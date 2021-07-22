import models
import os
from typing import List
from models.preprocessing import vocabulary

v = vocabulary.WordVocab()

text = []
with open(os.getcwd()+r"/data/sentences.txt", "r") as file:
    for line in file:
        if line == "\n":
            continue
        text.append(line)

v.fit_texts(text)
sequence = v.sentences2sequences(text)

print(sequence)