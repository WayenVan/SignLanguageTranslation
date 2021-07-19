import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from typing import List


class Vocabulary:

    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^å`{|}~\t\n'):
        self.tokenizer = Tokenizer(filters=filters)

    def fit_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def load(self, file_path):
        json_string = self.tokenizer.to_json()
        with open(file_path, "w+") as file:
            file.write(json_string)

    def save(self, file_path):
        with open(file_path, "r") as file:
            json_string = file.readline()
        self.tokenizer = tokenizer_from_json(json_string)

    def get_dictionary(self):
        return self.tokenizer.index_word, self.tokenizer.word_index


class GlossVocab(Vocabulary):

    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^å`{|}~\t\n'):
        super(GlossVocab, self).__init__(filters=filters)

    def glosses2index(self, glosses: List[List[str]]):
        """
        convert glosses into index
        :param glosses: list with a sequence of gloss name for each sentence
        :return: List[List[int]] index sequence of gloss
        """
        return [[self.tokenizer.word_index[gloss] for gloss in item] for item in glosses]

    def index2glosses(self, index: List[List[int]]):
        """
        conver index into glosses
        :param index: index sequence
        :return: List[List[str]] glosses sequence
        """
        return [[self.tokenizer.index_word[i] for i in item] for item in index]


class WordVocab(Vocabulary):

    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^å`{|}~\t\n'):
        super(WordVocab, self).__init__(filters=filters)

    def sentences2sequences(self, sentences):
        return self.tokenizer.texts_to_sequences(sentences)

    def sequences2sentences(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)
