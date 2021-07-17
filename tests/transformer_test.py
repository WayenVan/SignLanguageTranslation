import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from models.transformer.encoder import Encoder, PositionEmbedding

if __name__ == '__main__':
    attn = layers.MultiHeadAttention(num_heads=5, key_dim=24, value_dim=32)
    a = tf.ones(shape=(3, 5, 256))
    b = tf.ones(shape=(3, 5, 512))
    print(attn(a, b))