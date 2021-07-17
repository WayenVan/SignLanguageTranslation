import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import utils


class EncodeBlock(layers.Layer):

    def __init__(self, embed_dim, k_dim, v_dim, num_heads, ff_dim, rate=0.0):
        super(EncodeBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=k_dim, value_dim=v_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, input_mask=None, training=None, *args, **kwargs):
        """
        :param inputs: [batch_size, time_step, input_dim]
        :param input_mask: [batch_size, time_step_mask]
        :param training: if it returns
        :return: transformer_encode_output
        """

        # create mask and put it into attention
        if input_mask is not None:
            attn_mask = utils.create_att_mask(input_mask)
            attn_output = self.att(inputs, inputs, attention_mask=attn_mask, training=training)
        else:
            attn_output = self.att(inputs, inputs, training=training)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Encoder(layers.Layer):
    """
    transformer encoder, it include N endoerBlock.
    """

    def __init__(self, block_number, embed_dim, k_dim, v_dim,num_heads, ff_dim, rate=0.0):
        super(Encoder, self).__init__()
        self.encode_blocks = [
            EncodeBlock(embed_dim, k_dim, v_dim, num_heads, ff_dim, rate=rate) for i in range(block_number)
        ]

    def call(self, inputs, intput_mask=None, training=None, *args, **kwargs):
        output = inputs
        for block in self.encode_blocks:
            output = block(output, intput_mask=intput_mask, training=training)

        return output


class PositionEmbedding(layers.Layer):

    def __init__(self, d_model):
        super(PositionEmbedding, self).__init__()
        self._d_model = d_model

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: [batch_size, sequence_length, embed_dim]
        :return: tensor finishing positioin encode
        """
        position_code = utils.positional_encoding(inputs.shape[1], inputs.shape[-1])

        return inputs + position_code
