import json
import os
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

import models.losses as losses
import models.metrics as metrics

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

""""function"""
def eval_video2gloss(model: keras.Model, data_gen: DataGenerator, vocab: GlossVocab):
    data = data_gen._data
    glosses = [[sign["gloss"] for sign in item["signs"]]for item in data]
    gloss_pre = tf.concat([tf.argmax(model(item[0])[0], axis=-1) for item in data_gen], axis=0).numpy().tolist()
    gloss_pre = vocab.index2glosses(gloss_pre)

    output = {
        "hypothesis":glosses,
        "ground_trugh":gloss_pre
    }
    
    with open("./data/eval/glosses.json", "w+") as file:
        json.dump(output, file)



""" data preparation """
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:

    # data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"
    data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"
    with open(data_set_path + "/dataSet.json") as file:
        content = file.read()
        data = json.loads(content)

    data_length = len(data)
    word_vocab = WordVocab()
    gloss_vocab = GlossVocab()
    word_vocab.load(os.getcwd()+"/data/vcab/word_vocab.json")
    gloss_vocab.load(os.getcwd()+"/data/vcab/gloss_vocab.json")

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


    data_gen = DataGenerator(batch_size=1,
                             data_list=data,
                             gloss_dict=gloss_vocab,
                             word_dict=word_vocab,
                             sign_sequence_length=max_gloss_length,
                             sentence_length=max_sentence_length,
                             dataset_dir=data_set_path,
                             holistic=holistic,
                             mediate_output_dim=video_embed_dim,
                             gloss_only=False)

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

    opt = optimizers.SGD(learning_rate=1e-5)

    model.load_weights(os.getcwd() + "/data/training_data/checkpoint/checkpoint")
    model.trainable=False
    model.summary()

    #start evaluate

    bos_index = word_vocab.tokenizer.word_index["<bos>"]
    eos_inde = word_vocab.tokenizer.word_index["<eos>"]

    eval_video2gloss(model, data_gen, gloss_vocab)
