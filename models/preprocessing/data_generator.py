import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
from .mediapipe import pose_estimation
import numpy as np
from .vocabulary import GlossVocab, WordVocab
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import mediapipe as mp


class DataGenerator(Sequence):
    def __init__(self, batch_size,
                 data_list,
                 gloss_dict: GlossVocab,
                 word_dict: WordVocab,
                 sign_sequence_length,
                 sentence_length,
                 dataset_dir,
                 holistic,
                 mediate_output_dim,
                 frames_per_sign=5,
                 gloss_only=False):

        super(DataGenerator, self).__init__()
        self._batch_size = batch_size
        self._gloss_only = gloss_only
        self._data = data_list
        self._data_length = len(data_list)
        self._gloss_dict = gloss_dict
        self._word_dict = word_dict
        self._frames_per_sign = frames_per_sign
        self._sign_sequence_length = sign_sequence_length
        self._sentence_length = sentence_length
        self._dataset_dir = dataset_dir
        self._holistic = holistic
        self._mediate_output_dim = mediate_output_dim

    def __len__(self):
        return self._data_length // self._batch_size

    def __getitem__(self, index):
        start_index = index * self._batch_size
        video_batch = []
        mask_batch = []
        sentence_batch = []
        glosses_batch = []

        # generate
        for i in range(self._batch_size):
            video_clips, glosses, sentence = self._parse_sentence(self._data[start_index + i], self._holistic)
            assert video_clips.shape[0] <= self._sign_sequence_length

            # generate mask
            mask = np.ones(shape=self._sign_sequence_length, dtype=np.bool)
            if video_clips.shape[0] < self._sign_sequence_length:
                padding_length = self._sign_sequence_length - video_clips.shape[0]
                mask[-padding_length:] = False
                padding = tf.zeros(shape=(padding_length,
                                          video_clips.shape[1],
                                          video_clips.shape[2],
                                          video_clips.shape[3],
                                          video_clips.shape[4]), dtype=tf.float32)
                video_clips = tf.concat([video_clips, padding], axis=0)

            video_batch.append(video_clips)
            mask_batch.append(tf.cast(mask, dtype=tf.bool))
            sentence_batch.append(sentence)
            glosses_batch.append(glosses)

        # finish generating
        glosses_batch = self._gloss_dict.glosses2index(glosses_batch)
        glosses_batch = pad_sequences(glosses_batch, maxlen=self._sign_sequence_length, padding="post")

        sentence_input = ["<BOS> " + sentence for sentence in sentence_batch]
        sentence_output = [sentence + " <EOS>" for sentence in sentence_batch]

        sentence_input = self._word_dict.sentences2sequences(sentence_input)
        sentence_input = pad_sequences(sentence_input, maxlen=self._sentence_length + 1, padding="post")

        sentence_output = self._word_dict.sentences2sequences(sentence_output)
        sentence_output = pad_sequences(sentence_output, maxlen=self._sentence_length + 1, padding="post")

        #return input x and output y
        if self._gloss_only:
            return ([tf.stack(video_batch), tf.stack(mask_batch)],
                   [tf.stack(to_categorical(glosses_batch, num_classes=self._gloss_dict.get_token_num() + 1)),
                    tf.zeros(shape=(self._batch_size, self._sign_sequence_length, self._mediate_output_dim))])
        else:
            return [tf.stack(video_batch),
                    tf.stack(sentence_input),
                    tf.stack(mask_batch)], \
                   [to_categorical(tf.stack(glosses_batch), num_classes=self._gloss_dict.get_token_num() + 1),
                    to_categorical(tf.stack(sentence_output), num_classes=self._word_dict.get_token_num() + 1)]

    def get_glosses(self):
        return [[sign["gloss"] for sign in item["signs"]] for item in self._data]

    def get_sentences(self):
        return [item["sentence"] for item in self._data]

    def _parse_sentence(self, sentence, holistic):
        """
        generate list of video frames from a single sign
        """

        cap = cv2.VideoCapture()
        cap.open(self._dataset_dir + "/" + sentence["file_name"] + ".mp4")

        if not cap.isOpened():
            print("video open failed")
            exit(0)

        sign_clips = []
        glosses = []
        for sign in sentence["signs"]:
            glosses.append(sign["gloss"])

            sample_points = np.linspace(sign["position"][0], sign["position"][1], num=self._frames_per_sign,
                                        endpoint=True)
            sample_points = np.around(sample_points)

            frames = []
            for point in sample_points:
                # read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, point)
                result, frame = cap.read()
                assert result == True

                frame = pose_estimation(frame, holistic)
                frame = cv2.resize(frame, (256, 256))

                frames.append(tf.cast(frame, dtype=tf.float32))

            sign_clips.append(tf.stack(frames))

        return tf.stack(sign_clips), glosses, sentence["sentence"]
