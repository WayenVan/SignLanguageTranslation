from tensorflow.keras import losses
import tensorflow as tf

def cross_entrophy(y_actual, y_predicted):
    """
    :param y_actual: [batch_size, sequence_length, categorical]
    :param y_predicted: ([batch_size, sequence_length, categorical],
    :return:
    """

    y_pre = y_predicted
    y_pre = tf.reshape(y_pre, shape=(tf.shape(y_pre)[0]*tf.shape(y_pre)[1],
                                     tf.shape(y_pre)[-1]))
    y_actual = tf.reshape(y_actual, shape=(tf.shape(y_actual)[0]*tf.shape(y_actual)[1],
                                           tf.shape(y_actual)[-1]))

    return tf.math.reduce_sum(losses.kl_divergence(y_actual, y_pre))


if __name__ == '__main__':
    y1 = tf.constant([[[0., 0., 0., 1]]])
    y2 = tf.constant([[[0.2, 0.2, 0., 0.6]]])
    y3 = tf.constant([[[0.1, 0.1 ,0., 0.8]]])
    y4 = tf.constant([[[0.0, 0., 0., 1.]]])
    print(cross_entrophy(y1, y2))
    print(cross_entrophy(y1, y3))
    print(cross_entrophy(y1, y4))