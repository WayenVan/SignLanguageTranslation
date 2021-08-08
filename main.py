import tensorflow as tf
import tensorflow.keras as keras


tri = tf.ones(shape=(5, 5))  # [batch_size, q_dim, k_dim]
tri = tf.linalg.band_part(tri, -1, 0)
print(tri)