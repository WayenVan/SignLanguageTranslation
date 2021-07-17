import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization
from . import utils


class DecoderBlock(layers.Layer):
    """
    transformer decoder block in attention is all you need
    """

    def __init__(self, embed_dim, k_dim, v_dim, num_heads, ff_dim, rate=0.0):
        super(DecoderBlock, self).__init__()
        self.dropout1 = Dropout(rate=rate)
        self.dropout2 = Dropout(rate=rate)
        self.dropout3 = Dropout(rate=rate)

        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )

        self.masked_attn = MultiHeadAttention(num_heads=num_heads, key_dim=k_dim, value_dim=v_dim)
        self.cross_attn = MultiHeadAttention(num_heads=num_heads, key_dim=k_dim, value_dim=v_dim)

        self.LN1 = LayerNormalization()
        self.LN2 = LayerNormalization()
        self.LN3 = LayerNormalization()

    def call(self, inputs,
             inputs_mask=None,
             encoder_mask=None,
             training=None, *args, **kwargs):
        """
        :param inputs: a list with [decoder_inputs, encoder_outputs]
        :param inputs_mask: a tensor [batch_size, sequence_length]
        :param encoder_mask:  a tensor [batch_size, sequence_length]
        :param training: if the fucntion is called in during training
        """
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')

        attn_mask = utils.create_att_mask(inputs_mask, future_mask=True)
        ret1 = self.masked_attn(inputs[0], attention_mask=attn_mask)
        ret1 = self.dropout1(ret1, training=training)
        ret1 = self.LN1(ret1 + inputs[0])

        if inputs_mask is None:
            ret2 = self.cross_attn(ret1, inputs[1])
        else:
            cross_attn_mask = utils.create_att_mask(encoder_mask)
            ret2 = self.cross_attn(ret1, inputs[1], attention_mask=cross_attn_mask)

        ret2 = self.dropout2(ret2, training=training)
        ret2 = self.LN2(ret1 + ret2)

        ret3 = self.ffn(ret2)
        ret3 = self.dropout3(ret3, training=training)

        return self.LN3(ret3 + ret2)


class Decoder(layers.Layer):
    """
    decoder with num_block DecoderBlocks
    """

    def __init__(self,
                 num_block,
                 embed_dim,
                 k_dim,
                 v_dim,
                 num_heads,
                 ff_dim,
                 rate=0.0):
        super(Decoder, self).__init__()
        self.decoder_blocks = [
            DecoderBlock(embed_dim, k_dim, v_dim, num_heads, ff_dim, rate) for i in range(num_block)
        ]

    def call(self,
             inputs,
             inputs_mask=None,
             encoder_mask=None,
             training=None,
             *args, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')

        ret = inputs[0]
        encoder_outputs = inputs[1]

        for block in self.decoder_blocks:
            ret = block([ret, encoder_outputs],
                        inputs_mask=inputs_mask,
                        encoder_mask=encoder_mask,
                        training=training)
        return ret