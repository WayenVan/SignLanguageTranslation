import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self):
        super(DataGenerator, self).__init__()

    def __len__(self, data_list):
        pass

    def __getitem__(self, index):
        pass