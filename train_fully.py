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

video_input_shape=None #get from dataset
word_input_shape=None #get from dataset for max sentence sequence
video_embed_dim=256
word_embed_dim=256
gloss_categories=None #get from dataset
word_categories=None #get from dataset
num_block_encoder=6
num_block_decoder=6
head_num=12
k_dim=64
v_dim=64
ff_dim=2048
encoder_linear_hidden_dim=256
decoder_linear_hidden_dim=256
drop_out=0.1

""" data preparation """
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:

    data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"
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

    gloss_categories = len(gloss_vocab.get_dictionary()[0])
    word_categories = len(word_vocab.get_dictionary()[0])
    word_input_shape = max_sentence_length
    video_input_shape = (max_gloss_length, 5, 256, 256, 3)


    random.shuffle(data)

    data_gen = DataGenerator(batch_size=2,
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

    opt = optimizers.SGD(learning_rate=1e-4)
    model.load_weights(os.getcwd() + "/data/training_data/checkpoint/checkpoint")
    model.summary()
    model.compile(optimizer=opt,
                  loss=[losses.my_kl_divergence,
                        losses.my_kl_divergence],
                  loss_weights=[0.5, 1],
                  metrics=[metrics.my_categorical_accuracy])

    # prepare checkpoint
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.getcwd() + "/data/training_data/checkpoint/checkpoint",
        save_best_only=False,
        save_weights_only=True
    )

    history = model.fit(x=data_gen, verbose=1, epochs=20, callbacks=[checkpoint_callback])

    # save history
    with open(os.getcwd() + "/data/training_data/checkpoint/history", "w+") as file:
        hist = pd.DataFrame(history.history)
        hist.to_json(file)
