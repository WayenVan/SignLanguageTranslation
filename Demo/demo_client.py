import json
import os
import sys
import threading
sys.path.append(os.getcwd())

import cv2
import pickle

import mediapipe as mp
from models.preprocessing.data_generator import DataGenerator
from models.preprocessing.vocabulary import WordVocab, GlossVocab
import requests
import pyttsx3
from threading import Thread, Event
from queue import Queue
import typing

import gtts
from playsound import playsound

video_input_shape = None  # get from dataset
word_input_shape = None  # get from dataset for max sentence sequence
video_embed_dim = 256
word_embed_dim = 256
gloss_categories = 39  # get from dataset
word_categories = 46  # get from dataset
num_block_encoder = 6
num_block_decoder = 12
head_num = 12
k_dim = 64
v_dim = 64
ff_dim = 2048
encoder_linear_hidden_dim = 256
decoder_linear_hidden_dim = 256
drop_out = 0.1

q = Queue()
"""speaking thread"""
def speaking(que: Queue):
    while True:
        string = que.get(block=True)
        tts = gtts.gTTS(string)
        tts.save("Demo/template.mp3")
        playsound("Demo/template.mp3")


threading.Thread(target=speaking, args=(q,)).start()

"""main thread"""
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:
    # data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"
    data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"

    with open(data_set_path + "/dataSet.json") as file:
        content = file.read()
        data = json.loads(content)

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

    assert gloss_categories == len(gloss_vocab.get_dictionary()[0])
    assert word_categories == len(word_vocab.get_dictionary()[0])

    data_gen = DataGenerator(batch_size=1,
                             data_list=data,
                             gloss_dict=gloss_vocab,
                             word_dict=word_vocab,
                             sign_sequence_length=max_gloss_length,
                             sentence_length=max_sentence_length,
                             dataset_dir=data_set_path,
                             holistic=holistic,
                             mediate_output_dim=video_embed_dim,
                             gloss_only=True)

    """playing and send"""
    for index, data in enumerate(data_gen):

        file_name = data_gen._data[index]["file_name"]
        cap = cv2.VideoCapture()
        cap.open(data_set_path + "/" + file_name + ".mp4")

        # play the video
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()

        s = pickle.dumps(data[0])
        #response = requests.post("http://172.30.148.9:2333", data=s)
        q.put("my favourite number is 0")
