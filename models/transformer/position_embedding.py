import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionEmbedding(layers.Layer):

    def __init__(self, d_model, max_len=100):
        super(PositionEmbedding, self).__init__()
        self._d_model = d_model
        self._max_len = max_len
        self._positional_code = self.positional_encoding(max_len, d_model)

    def call(self, inputs, *args, **kwargs):
        return inputs + self._positional_code[:, :tf.shape(inputs)[1], :]

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
