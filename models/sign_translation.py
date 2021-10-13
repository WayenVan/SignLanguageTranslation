import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, Dropout
from models.video2gloss import create_video2gloss_model, Video2Gloss
from models.transformer.decoder import Decoder
from models.transformer.position_embedding import PositionEmbedding


class SignTranslation(keras.Model):

    def get_config(self):
        pass

    def __init__(self,
                 video_embed_dim,
                 word_embed_dim,
                 gloss_categories,
                 word_categories,
                 num_block_encoder,
                 num_block_decoder,
                 head_num,
                 k_dim,
                 v_dim,
                 ff_dim,
                 encoder_linear_hidden_dim,
                 decoder_linear_hidden_dim,
                 drop_out=0.0,
                 ):
        super(SignTranslation, self).__init__()

        # save parameter
        self._video_embed_dim = video_embed_dim
        self._word_embed_dim = word_embed_dim
        self._gloss_categories = gloss_categories
        self._word_categories = word_categories
        self._num_block_encoder = num_block_encoder
        self._num_block_decoder = num_block_decoder
        self._head_num = head_num
        self._k_dim = k_dim
        self._v_dim = v_dim
        self._ff_dim = ff_dim
        self._encoder_linear_hidden_dim = encoder_linear_hidden_dim
        self._decoder_linear_hidden_dim = decoder_linear_hidden_dim
        self._drop_out = drop_out

        # create models and layers
        self.video2gloss = Video2Gloss(video_embed_dim=video_embed_dim,
                                       block_number=num_block_encoder,
                                       k_dim=k_dim,
                                       v_dim=v_dim,
                                       encoder_head_number=head_num,
                                       ff_dim=ff_dim,
                                       linear_hidden_dim=encoder_linear_hidden_dim,
                                       linear_output_dim=gloss_categories + 1,
                                       drop_out=drop_out)

        self.decoder = Decoder(num_block=num_block_decoder,
                               embed_dim=word_embed_dim,
                               k_dim=k_dim,
                               v_dim=v_dim,
                               num_heads=head_num,
                               ff_dim=ff_dim,
                               rate=drop_out)

        self.position_embed = PositionEmbedding(word_embed_dim)
        self.word_embed = Embedding(input_dim=word_categories + 1, output_dim=word_embed_dim, mask_zero=True)
        self.linear = keras.models.Sequential([
            Dense(decoder_linear_hidden_dim, activation='relu'),
            Dropout(rate=drop_out),
            Dense(word_categories + 1, activation="softmax")
        ])

    def load_video2gloss(self, path):
        self.video2gloss.load_weights(path)

    def call(self, inputs, training=None, *args, **kwargs):
        """

        :param inputs: List of tensors tpye [video_inputs, word_inputs, video_mask] where:
        video_inputs: a tensor of [batch_size, sequence_length, frame_number, ...(frame dimension)]
        video_mask: a boolean or int tensor for masking [batch_size, sequence_length]
        word_inputs: a pre-padding tensor of [batch_size, sequence_length(index of token)]
        :param training: if the model is training
        :return: gloss_output, word_output
        """
        video_mask = inputs[2]
        video_inputs = inputs[0]
        word_inputs = inputs[1]

        word_mask = self.word_embed.compute_mask(word_inputs)
        word_embed = self.word_embed(word_inputs)
        word_embed = self.position_embed(word_embed)

        gloss_output, encoder_output = self.video2gloss([video_inputs, video_mask], training=training)
        decoder_output = self.decoder([word_embed, encoder_output, word_mask, video_mask], training=training)
        word_output = self.linear(decoder_output, training=training)

        return gloss_output, word_output

    def iterative_prediction(self, inputs, bos_index, eos_index):
        """
        :param inputs: [input_data:[1, sequence_length, frame_size, frame_height,
                        frame_width, channel], mask:[1, sequence_length]]
        :return: a list of [sequence_length] with the predicted word index
        """
        video_inputs = inputs[0]
        video_mask = inputs[1]
        _, encoder_output = self.video2gloss([video_inputs, video_mask])

        ret = []
        word_inputs = tf.constant([[bos_index]], dtype=tf.int64)

        for i in range(self._max_iterate_num):

            word_mask = self.word_embed.compute_mask(word_inputs)
            word_embed = self.word_embed(word_inputs)
            word_embed = self.position_embed(word_embed)

            decoder_output = self.decoder([word_embed, encoder_output, word_mask, video_mask])
            word_output = self.linear(decoder_output)
            words_pre = tf.argmax(word_output, axis=-1)

            if words_pre[0][i] == self._eos_index:
                ret = tf.squeeze(words_pre, axis=0).numpy().tolist()
                break

            if i == self._max_iterate_num - 1:
                print("warning, out of the range!")
                ret = tf.squeeze(words_pre, axis=0).numpy().tolist()
                break

            word_inputs = tf.concat([tf.constant([[self._bos_index]], dtype=tf.int64), words_pre], axis=1)

        return ret


def create_sign_translation_model(video_input_shape,
                                  word_input_shape,
                                  video_embed_dim,
                                  word_embed_dim,
                                  gloss_categories,
                                  word_categories,
                                  num_block_encoder,
                                  num_block_decoder,
                                  head_num,
                                  k_dim,
                                  v_dim,
                                  ff_dim,
                                  encoder_linear_hidden_dim,
                                  decoder_linear_hidden_dim,
                                  drop_out=0.0):

    model = SignTranslation(video_embed_dim=video_embed_dim,
                            word_embed_dim=word_embed_dim,
                            gloss_categories=gloss_categories,
                            word_categories=word_categories,
                            num_block_encoder=num_block_encoder,
                            num_block_decoder=num_block_decoder,
                            head_num=head_num,
                            k_dim=k_dim,
                            v_dim=v_dim,
                            ff_dim=ff_dim,
                            encoder_linear_hidden_dim=encoder_linear_hidden_dim,
                            decoder_linear_hidden_dim=decoder_linear_hidden_dim,
                            drop_out=drop_out)

    video_input = keras.Input(shape=video_input_shape)
    word_input = keras.Input(shape=word_input_shape, dtype=tf.int64)
    video_mask_input = keras.Input(shape=video_input_shape[0], dtype=tf.int8)

    # feed fake data
    _, _ = model([video_input, word_input, video_mask_input])

    return model
