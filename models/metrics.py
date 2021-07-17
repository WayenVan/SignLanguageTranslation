import tensorflow as tf
import tensorflow.keras as keras
def my_categorical_accuracy(y_actual, y_predicted):
    """

    :param y_true: [batch_size, sequence_length, categories]
    :param y_predict: [batch_size, sequence_length, categories]
    :return:
    """
    #reshape into [batch_size*sequence_length, categories]
    y_pre = tf.reshape(y_predicted, shape=(tf.shape(y_predicted)[0]*tf.shape(y_predicted)[1],
                                           tf.shape(y_predicted)[-1]))
    y_act = tf.reshape(y_actual, shape=(tf.shape(y_actual)[0]*tf.shape(y_actual)[1],
                                        tf.shape(y_actual)[-1]))

    return keras.metrics.categorical_accuracy(y_act, y_pre)