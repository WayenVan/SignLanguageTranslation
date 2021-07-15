import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
from models.transformer.encoder import Encoder, PositionEmbedding

if __name__ == '__main__':
    model = Encoder(block_number=6, embed_dim=400, num_heads=6, ff_dim=300)
    position_embed = PositionEmbedding(d_model=400)
    x = tf.ones(shape=(5, 5, 400))
    mask = tf.ones(shape=(5, 5))
    y = model(x, input_mask=mask)
    print(y.shape())