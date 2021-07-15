
import sys
import os

sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from models import losses as self_defined_losses
import models

from models.video2gloss import create_video2gloss_model


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train = tf.image.per_image_standardization(
    #     x_train
    # )
    # x_train = tf.reshape(x_train, shape=(10000, 6, 4, 7, 28, 1))

    # y_train = to_categorical(y_train, num_classes=10)
    # y_train = tf.reshape(y_train, shape=(10000, 6, 10))
    # blank = tf.zeros(shape=(10000, 6, 512))

    
    x = tf.random.uniform(shape=(8, 6, 4, 7, 28, 1))
    y = tf.random.uniform(shape=(8, 6, 10))
    blank = tf.zeros(shape=(8, 6, 512))
    model = create_video2gloss_model(input_shape=(6, 4, 7, 28, 1))
    adam = optimizers.Adam(learning_rate=10e-4)
    
    model.summary()
    model.compile(optimizer=adam, loss=[self_defined_losses.cross_entrophy,
                                          self_defined_losses.cross_entrophy],
                                    loss_weights=[1, 0])

    model.fit(x=x, y=[y, blank], batch_size=32, epochs=3, verbose=1)