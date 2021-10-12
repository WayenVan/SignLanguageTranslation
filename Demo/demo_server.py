import json
import os

import cv2
import pandas as pd
import numpy as np

import mediapipe as mp
from tensorflow.python.keras.engine.training import concat

from models.video2gloss import create_video2gloss_model
from models.sign_translation import create_sign_translation_model

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
import random
from models.preprocessing.data_generator import DataGenerator
from models.preprocessing.vocabulary import WordVocab, GlossVocab


video_input_shape=None #get from dataset
word_input_shape=None #get from dataset for max sentence sequence
video_embed_dim=256
word_embed_dim=256
gloss_categories=39 #get from dataset
word_categories=46 #get from dataset
num_block_encoder=6
num_block_decoder=12
head_num=12
k_dim=64
v_dim=64
ff_dim=2048
encoder_linear_hidden_dim=256
decoder_linear_hidden_dim=256
drop_out=0.1

"""prepare data"""
data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"
# data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"
with open(data_set_path + "/dataSet.json") as file:
    content = file.read()
    data = json.loads(content)

data_length = len(data)
word_vocab = WordVocab()
gloss_vocab = GlossVocab()
word_vocab.load(os.getcwd() + "/data/vcab/word_vocab.json")
gloss_vocab.load(os.getcwd() + "/data/vcab/gloss_vocab.json")

# calculate max length
max_gloss_length = 0
max_sentence_length = 0

for item in data:
    glosses = [sign["gloss"] for sign in item["signs"]]
    if len(glosses) > max_gloss_length:
        max_gloss_length = len(glosses)

    sentence_length = len(word_vocab.sentences2sequences([item["sentence"]])[0])
    if sentence_length > max_sentence_length:
        max_sentence_length = sentence_length

print("data length: ", len(data))
print(gloss_vocab.get_dictionary()[0])
print(word_vocab.get_dictionary()[0])

assert gloss_categories == len(gloss_vocab.get_dictionary()[0])
assert word_categories == len(word_vocab.get_dictionary()[0])
word_input_shape = max_sentence_length
video_input_shape = (max_gloss_length, 5, 256, 256, 3)

class ModelServer(keras.Model):
    """
    a wrapper for recursive generation of input
    """
    def __init__(self, model, bos_index, eos_index, max_iterate_num=15):
        super(ModelServer, self).__init__()
        self._model = model
        self._bos_index = bos_index
        self._eos_index = eos_index
        self._max_iterate_num = max_iterate_num

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [[1, sequence_length, frame_size, frame_height, frame_width, channel], [1, sequence_length]]
        :return ret: [sequence_length] a sequence of the index of predicted words
        """
        for i in range(self._max_iterate_num):
            output = model([inputs[0], text_input, inputs[1]])
            words_pre = tf.argmax(output[1], axis=-1)

            if words_pre[0][i] == self._eos_index:
                ret = tf.squeeze(words_pre, axis=0)
                break

            if i == self._max_iterate_num - 1:
                print("warning, out of the range!")
                ret = tf.squeeze(words_pre, axis=0)
                break

            text_input = tf.concat([tf.constant([[self._bos_index]], dtype=tf.int64), words_pre], axis=1)

        return ret




model = create_sign_translation_model(video_input_shape=video_input_shape,
                                      word_input_shape=word_input_shape,
                                      video_embed_dim=video_embed_dim,
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

model.load_weights(os.getcwd() + "/data/training_data/checkpoint/checkpoint")
model.trainable=False
model.summary()

bos_index = word_vocab.tokenizer.word_index["<bos>"]
eos_index = word_vocab.tokenizer.word_index["<eos>"]

model_server = ModelServer(model, bos_index=bos_index, eos_index=eos_index)

tf.placeholder