import tensorflow as tf
import tensorflow.keras as keras

l = keras.layers.Embedding(100, 200, mask_zero=True)
p = keras.layers.Dense(64, activation="softmax")

x = tf.ones(shape=(20, 100))
y = tf.zeros(shape=(1, 100))
x = tf.concat([x, y], axis=0)
x = l(x)
x = p(x)

print(x)