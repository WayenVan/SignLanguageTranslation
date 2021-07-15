import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer, Dense, Dropout

from .cnn.resnet3D import ReNet3D18L
from .transformer.encoder import Encoder, PositionEmbedding

from tensorflow.keras.utils import to_categorical

from . import losses as self_defined_losses

class Video2Gloss(layers.Layer):

    def __init__(self,
                 video_embed_dim,
                 block_number,
                 encoder_head_number,
                 ff_dim,
                 linear_hidden_dim,
                 linear_output_dim,
                 drop_out=0.0):
        super(Video2Gloss, self).__init__()

        # record hyper parameter
        self._video_embed_dim = video_embed_dim
        self._block_number = block_number
        self._encoder_head_number = encoder_head_number
        self._ff_dim = ff_dim
        self._linear_hidden_dim = linear_hidden_dim
        self._linear_output_dim = linear_output_dim

        # 3d cnn
        self.resnet_3d = ReNet3D18L(video_embed_dim)
        # position encode
        self.position_embedding = PositionEmbedding(video_embed_dim)
        # transformer encode:
        self.encoder = Encoder(block_number,
                               video_embed_dim,
                               encoder_head_number,
                               ff_dim,
                               rate=drop_out)
        # classify layer:
        self.linear = keras.models.Sequential([
            Dense(linear_hidden_dim, activation="relu"),
            Dropout(rate=drop_out),
            Dense(linear_output_dim, activation="softmax"),
        ])

    def get_config(self):
        pass

    def call(self, inputs, input_mask=None, training=None, *args, **kwargs):
        """

        :param inputs:
        :param input_mask:
        :param training:
        :param args:
        :param kwargs:
        :return: (gloss classification [batch_size, sequence_length, vocabulary_size],
        video_feature [batch_size, sequence_length, ff_dim])
        """
        video_feature = self.resnet_3d(inputs)
        video_feature = self.position_embedding(video_feature)
        video_feature = self.encoder(video_feature, input_mask=input_mask, training=training)
        gloss_category = self.linear(video_feature, training=training)

        return gloss_category, video_feature


def create_video2gloss_model(input_shape, ):
    video2gloss = Video2Gloss(video_embed_dim=512,
                              block_number=6,
                              encoder_head_number=6,
                              ff_dim=256,
                              linear_hidden_dim=256,
                              linear_output_dim=10,
                              drop_out=0.1)

    inputs = keras.Input(shape=input_shape) #feed fake batch_size for timedistributed computine
    gloss_output, video_feature_output = video2gloss(inputs)
    model = keras.Model(inputs=inputs, outputs=[gloss_output, video_feature_output])

    return model

def save_video2gloss_model():
    pass
