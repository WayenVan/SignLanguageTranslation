import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, Dropout
from models.video2gloss import create_video2gloss_model
from models.transformer.decoder import Decoder
from models.transformer.encoder import PositionEmbedding

class SignTranslation(layers.Layer):

    def __init__(self,
                 video_input_shape,
                 video_embed_dim,
                 word_embed_dim,
                 gloss_categories,
                 word_categories,
                 num_block_encoder,
                 num_block_decoder,
                 head_num,
                 k_dim,
                 v_dim,
                 ff_dim,
                 encoder_linear_hidden_dim,
                 decoder_linear_hidden_dim,
                 drop_out=0.0,
                 load_video2gloss_parameter=False,
                 video2gloss_path=None
                 ):
        super(SignTranslation, self).__init__()
        self.video2gloss = create_video2gloss_model(input_shape=video_input_shape,
                                                    video_embed_dim=video_embed_dim,
                                                    block_number=num_block_encoder,
                                                    k_dim=k_dim,
                                                    v_dim=v_dim,
                                                    encoder_head_number=head_num,
                                                    ff_dim=ff_dim,
                                                    linear_hidden_dim=encoder_linear_hidden_dim,
                                                    linear_output_dim=gloss_categories+1,
                                                    drop_out=drop_out)

        self.decoder = Decoder(num_block=num_block_decoder,
                               embed_dim=word_embed_dim,
                               k_dim=k_dim,
                               v_dim=v_dim,
                               num_heads=head_num,
                               ff_dim=ff_dim,
                               rate=drop_out)

        self.position_embed = PositionEmbedding(word_embed_dim)
        self.word_embed = Embedding(input_dim=word_categories+1, output_dim=word_embed_dim, mask_zero=True)
        self.linear = keras.models.Sequential([
            Dense(decoder_linear_hidden_dim, activation='relu'),
            Dropout(rate=drop_out),
            Dense(word_categories+1, activation="softmax")
        ])

        #load parameter
        if load_video2gloss_parameter:
            if video2gloss_path == None:
                raise("please specify video2gloss_path parameter when load_video2gloss_parameter is True")
            else:
                self.video2gloss.load_weights(video2gloss_path)

    def call(self, inputs, training=None, *args, **kwargs):
        """

        :param inputs: List of tensors tpye [video_inputs, word_inputs, video_mask] word_inputs
        should be a tensor of [batch_size, sequence_length] which include the indexes
        of each sequence. This sequece should be padding with 0.
        :param video_mask: a boolean tensor of [batch_size, sequence_length]
        :param word_mask: a boolean tensor of [batch_size, sequence_length]
        :param training: if the model is training
        :return: gloss_output, word_output
        """
        video_mask = inputs[2]
        video_inputs = inputs[0]
        word_inputs = inputs[1]

        word_mask =  self.word_embed.compute_mask(word_inputs)
        word_embed = self.word_embed(word_inputs)
        word_embed = self.position_embed(word_embed)

        gloss_output, encoder_output = self.video2gloss([video_inputs, video_mask], training=training)
        decoder_output = self.decoder([word_embed, encoder_output, word_mask, video_mask], training=training)
        word_output = self.linear(decoder_output, training=training)

        return gloss_output, word_output

