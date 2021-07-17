import sys
import os
import pickle

sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from models import losses
from models import metrics
import models

from models.video2gloss import create_video2gloss_model


def prepare_data():
    """

    :return: 10000 sample of train, 1000 sample of test
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, newshape=(60000, 28, 28, 1))
    x_train = tf.image.per_image_standardization(x_train)
    x_train = tf.reshape(x_train, shape=(10000, 6, 4, 7, 28, 1))

    y_train = to_categorical(y_train, num_classes=10)
    y_train = tf.reshape(y_train, shape=(10000, 6, 10))
    blank = tf.zeros(shape=(10000, 6, 128))

    x_test = x_test[:6000]
    x_test = np.reshape(x_test, newshape=(6000, 28, 28, 1))
    x_test = tf.image.per_image_standardization(x_test)
    x_test = tf.reshape(x_test, shape=(1000, 6, 4, 7, 28, 1))

    y_test = y_test[:6000]
    y_test = to_categorical(y_test, num_classes=10)
    y_test = tf.reshape(y_test, shape=(1000, 6, 10))

    return (x_train, y_train), (x_test, y_test)


def train(model, x_train, y1_train, y2_train):
    # prepare checkpoint
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.getcwd() + "/data/checkpoint",
        save_weights_only=True,
    )

    history = model.fit(x=x_train[:500], y=[y1_train[:500], y2_train[:500]],
                        batch_size=4,
                        epochs=50,
                        verbose=1,
                        callbacks=[checkpoint_callback,])
    #save history
    with open(os.getcwd() + "/data/history", "wb+") as file:
        pickle.dump(history, file)

def evaluation(model, x_test, y_test):
    pass


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = prepare_data()
    blank = tf.zeros(shape=(10000, 6, 128))

    # create model
    model = create_video2gloss_model(input_shape=(6, 4, 7, 28, 1),
                                     video_embed_dim=128,
                                     block_number=6,
                                     encoder_head_number=6,
                                     ff_dim=64,
                                     linear_hidden_dim=256,
                                     linear_output_dim=10,
                                     drop_out=0.1)

    opt = optimizers.SGD()
    model.summary()
    model.compile(optimizer=opt,
                  loss=[losses.my_kl_divergence,
                        losses.my_kl_divergence],
                  loss_weights=[1, 0],
                  metrics=[metrics.my_categorical_accuracy])

    #if load weight
    model.load_weights(os.getcwd() + "/data/checkpoint")
    model.evaluate(x_test[:100], [y_test[:100], blank[:100]])
