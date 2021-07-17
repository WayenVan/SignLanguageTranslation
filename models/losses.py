from tensorflow.keras import losses
import tensorflow as tf


def my_kl_divergence(y_actual, y_predicted):
    """
    :param y_actual: [batch_size, sequence_length, categories]
    :param y_predicted: ([batch_size, sequence_length, catecogires],
    :return: sum value of all losses in batch
    """
    # reshape into [batch_size*sequence_length, categories]
    y_pre = tf.reshape(y_predicted, shape=(tf.shape(y_predicted)[0] * tf.shape(y_predicted)[1],
                                           tf.shape(y_predicted)[-1]))
    y_act = tf.reshape(y_actual, shape=(tf.shape(y_actual)[0] * tf.shape(y_actual)[1],
                                        tf.shape(y_actual)[-1]))

    return tf.math.reduce_sum(losses.kl_divergence(y_act, y_pre))
