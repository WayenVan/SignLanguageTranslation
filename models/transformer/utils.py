import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_att_mask(input_mask, future_mask=False):
    """
    :param mask: boolean tensor [batch_size, time_step_mask]
    :param future_mask: if mask future when do self attention vectors
    :return: [batch_size, time_step_mask, time_step_mask]
    """
    if future_mask:
        # create triangle matrix by numpy
        tri = np.ones(shape=(1, input_mask.shape[-1], input_mask.shape[-1]))
        tri = np.tril(tri, 0)
        att_mask = tf.constant(tri, dtype=tf.int8)
        assert len(att_mask.shape) == 3
        assert att_mask.shape[0] == 1
    else:
        att_mask = tf.ones(shape=(1, input_mask.shape[-1], input_mask.shape[-1]), dtype=tf.int8)
    # align the input mask with attention mask
    att_mask = tf.where(tf.expand_dims(input_mask, -2), att_mask, 0)  # mask become: [batch_size, 1, time_step]
    return att_mask


def positional_encoding(position, d_model):
    """
    position encoding using sine and cosine
    :param position: sequence_length
    :param d_model: demention of embedding
    :return: position encode
    """
    #define get angles
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

if __name__ == '__main__':
    a = tf.zeros(shape=(2, 3))
    print(tf.expand_dims(a, -2).shape.as_list())
    mask = tf.constant([[True, True, False, False, False],
                        [True, True, True, False, False]], dtype=tf.bool)

    print(mask)
    print(create_att_mask(mask))

