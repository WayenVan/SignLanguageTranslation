import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import numpy as np

from models.sign_translation import create_sign_translation_model


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


if __name__ == '__main__':
    model = create_sign_translation_model(video_input_shape=(6, 4, 7, 28, 1),
                                          word_input_shape=6,
                                          video_embed_dim=128,
                                          word_embed_dim=128,
                                          gloss_categories=9,
                                          word_categories=9,
                                          num_block_encoder=6,
                                          num_block_decoder=6,
                                          head_num=12,
                                          k_dim=64,
                                          v_dim=64,
                                          ff_dim=1024,
                                          encoder_linear_hidden_dim=256,
                                          decoder_linear_hidden_dim=256,
                                          drop_out=0.1)

    model.summary()