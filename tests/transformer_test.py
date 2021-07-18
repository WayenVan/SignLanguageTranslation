import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from models.transformer.encoder import Encoder, PositionEmbedding
from models.transformer.decoder import Decoder

if __name__ == '__main__':
    model = Decoder(6, 256, 64, 64, 12, 1024, rate=0.1)
    x = [tf.ones(shape=(5, 7, 256)), tf.ones(shape=(5, 4, 128))]
    input_mask = tf.ones(shape=(5, 7), dtype=tf.bool)
    encoder_mask =tf.ones(shape=(5, 4), dtype=tf.bool)
    y = model(x, inputs_mask=input_mask, encoder_mask=encoder_mask)
    print(y)