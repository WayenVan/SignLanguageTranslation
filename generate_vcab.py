import sys
import os

sys.path.append(os.getcwd())
from models.preprocessing.vocabulary import WordVocab, GlossVocab
import json


data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"

with open(data_set_path + "/dataSet.json") as file:
    content = file.read()
    data = json.loads(content)

data_length = len(data)
word_vocab = WordVocab()
gloss_vocab = GlossVocab()


for item in data:
    glosses = [sign["gloss"] for sign in item["signs"]]

    gloss_vocab.fit_texts([glosses])
    word_vocab.fit_texts([item["sentence"]])

# fit special tokens
word_vocab.fit_texts([["<BOS>", "<EOS>"]])

print("data length: ", len(data))
print(gloss_vocab.get_dictionary()[0])
print(word_vocab.get_dictionary()[0])

word_vocab.save(os.getcwd()+"/data/vcab/word_vocab.json")
gloss_vocab.save(os.getcwd()+"/data/vcab/gloss_vocab.json")