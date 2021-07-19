import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from models.transformer.decoder import Decoder

if __name__ == '__main__':
    layer = Decoder(6, 256, 64, 64, 12, 1024, rate=0.1)
    x1 = keras.Input(shape=(7, 256))
    x2 = keras.Input(shape=(4, 128))
    input_mask = keras.Input(shape=7, dtype=tf.bool)
    encoder_mask = keras.Input(shape=4, dtype=tf.bool)
    y = layer([x1, x2], inputs_mask=input_mask, encoder_mask=encoder_mask)
    print(y)

    model = keras.Model(inputs=[x1, x2, input_mask, encoder_mask], outputs=[y])
    model.summary()
