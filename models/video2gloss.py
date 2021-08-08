import sys

sys.path.append("..")

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout

from .cnn.resnet3D import ReNet3D18L
from .transformer.encoder import Encoder
from .transformer.position_embedding import PositionEmbedding


class Video2Gloss(keras.Model):

    def __init__(self,
                 video_embed_dim,
                 block_number,
                 k_dim,
                 v_dim,
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
                               k_dim,
                               v_dim,
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

    def call(self, inputs, training=None, *args, **kwargs):
        """
        :param inputs: a list of [inputs_tensor, inputs_mask]
        :param training: if the model is training or not
        :return: (gloss classification [batch_size, sequence_length, vocabulary_size],
        video_feature [batch_size, sequence_length, ff_dim])
        """
        assert len(inputs) == 2

        input_mask = inputs[1]
        inputs = inputs[0]

        video_feature = self.resnet_3d(inputs)
        video_feature = self.position_embedding(video_feature)
        video_feature = self.encoder([video_feature, input_mask], training=training)
        gloss_category = self.linear(video_feature, training=training)

        return gloss_category, video_feature


def create_video2gloss_model(input_shape,
                             video_embed_dim,
                             block_number,
                             k_dim,
                             v_dim,
                             encoder_head_number,
                             ff_dim,
                             linear_hidden_dim,
                             linear_output_dim,
                             drop_out=0.1
                             ):
    # create layer
    video2gloss = Video2Gloss(video_embed_dim=video_embed_dim,
                              block_number=block_number,
                              k_dim=k_dim,
                              v_dim=v_dim,
                              encoder_head_number=encoder_head_number,
                              ff_dim=ff_dim,
                              linear_hidden_dim=linear_hidden_dim,
                              linear_output_dim=linear_output_dim,
                              drop_out=drop_out)

    inputs_mask = keras.Input(shape=input_shape[0])
    inputs = keras.Input(shape=input_shape)

    # feed fake batch_size for Redistributed computing
    _, _ = video2gloss([inputs, inputs_mask])

    return video2gloss


def save_video2gloss_model():
    pass
