import json
import os
import pandas as pd

import mediapipe as mp

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

""" data preparation """
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:

    data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"
    with open("data/Dataset/Data/dataSet.json") as file:
        content = file.read()
        data = json.loads(content)

    data_length = len(data)
    word_vocab = WordVocab()
    gloss_vocab = GlossVocab()

    max_gloss_length = 0
    max_sentence_length = 0

    for item in data:
        glosses = [sign["gloss"] for sign in item["signs"]]
        if len(glosses) > max_gloss_length:
            max_gloss_length = len(glosses)

        gloss_vocab.fit_texts([glosses])
        word_vocab.fit_texts([item["sentence"]])

        sentence_length = len(word_vocab.sentences2sequences([item["sentence"]])[0])
        if sentence_length > max_sentence_length:
            max_sentence_length = sentence_length

    # fit special tokens
    word_vocab.fit_texts([["<BOS>", "<EOS>"]])

    print("data length: ", len(data))
    print(gloss_vocab.get_dictionary()[0])
    print(word_vocab.get_dictionary()[0])

    random.shuffle(data)

    data_gen = DataGenerator(batch_size=2,
                             data_list=data,
                             gloss_dict=gloss_vocab,
                             word_dict=word_vocab,
                             sign_sequence_length=max_gloss_length,
                             sentence_length=max_sentence_length,
                             dataset_dir=data_set_path,
                             holistic=holistic,
                             mediate_output_dim=256,
                             gloss_only=True)

    model = create_video2gloss_model(input_shape=(max_gloss_length, 5, 256, 256, 3),
                                     video_embed_dim=256,
                                     block_number=6,
                                     k_dim=64,
                                     v_dim=64,
                                     encoder_head_number=12,
                                     ff_dim=2048,
                                     linear_hidden_dim=256,
                                     linear_output_dim=gloss_vocab.get_token_num() + 1,
                                     drop_out=0.1)

    opt = optimizers.SGD(learning_rate=1e-5)
    model.summary()
    model.compile(optimizer=opt,
                  loss=[losses.my_kl_divergence,
                        losses.my_kl_divergence],
                  loss_weights=[1, 0],
                  metrics=[metrics.my_categorical_accuracy])

    # prepare checkpoint
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.getcwd() + "/data/Template/checkpoint",
        save_best_only=False,
        save_weights_only=True
    )

    history = model.fit(x=data_gen, verbose=1, epochs=4)

    # save history
    with open(os.getcwd() + "/data/Template/history", "w+") as file:
        hist = pd.DataFrame(history.history)
        hist.to_json(file)