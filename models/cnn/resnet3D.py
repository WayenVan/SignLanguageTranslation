import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Conv3D, Activation, \
    MaxPool3D, Dense, AvgPool3D, Flatten, \
    TimeDistributed, Dropout


# 18-layer architecture of 3d residual network
class ReNet3D18L(layers.Layer):

    def __init__(self, output_dim):
        super(ReNet3D18L, self).__init__()
        self.conv1 = Conv3D(64, 7, strides=(1, 2, 2), padding="same")
        self.max_pool = TimeDistributed(
            MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same"))

        self.conv2_x = [ResBlock3D(64, 64), ResBlock3D(64, 64)]
        self.conv3_x = [ResBlock3D(64, 128), ResBlock3D(128, 128)]
        self.conv4_x = [ResBlock3D(128, 256), ResBlock3D(256, 256)]
        self.conv5_x = [ResBlock3D(256, 512), ResBlock3D(512, 512)]

        self.average_pool = TimeDistributed(
            AvgPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same"))
        self.flatten = TimeDistributed(
            Flatten())

        self.fc = Dense(output_dim, activation="relu")

    def call(self, inputs, training=None, *args, **kwargs):
        # input: [batch_size, time_step, frame_size, ...(picture's dimension)]

        res_x = self.conv1(inputs)
        res_x = self.max_pool(res_x)

        # calculate each residual blocks
        for block in self.conv2_x + self.conv3_x + self.conv4_x + self.conv5_x:
            res_x = block(res_x)

        res_x = self.average_pool(res_x)
        res_x = self.flatten(res_x)
        # after flatten the x should be [batch_size, time_step, vector-dim]
        res_x = self.fc(res_x)

        return res_x


# ResBlock layer for sequence model:
class ResBlock3D(layers.Layer):

    def __init__(self, input_filter, output_filter,
                 kernel_size=3):
        super(ResBlock3D, self).__init__()

        self.input_filter = input_filter
        self.output_filter = output_filter
        self.kernel_size = kernel_size

        self.LN1 = LayerNormalization()
        self.LN2 = LayerNormalization()

        self.ReLU1 = Activation('relu')
        self.ReLU2 = Activation('relu')

        self.conv1 = Conv3D(output_filter, kernel_size, strides=1, padding="same")
        self.conv2 = Conv3D(output_filter, kernel_size, strides=1, padding="same")

        # if the output_filter!=input_filter, we need a convolutional layer to transform them into the same depth
        self.ConV_connect = Conv3D(output_filter, 1, strides=1, padding="same")

    def call(self, inputs, *args, **kwargs):
        # inputs: [batch_size, time_step, frame_size, ....(dim of picture)]

        assert inputs.shape[-1] == self.input_filter

        res_x = self.LN1(inputs)
        res_x = self.ReLU1(res_x)
        res_x = self.conv1(res_x)
        res_x = self.LN2(res_x)
        res_x = self.ReLU2(res_x)
        res_x = self.conv2(res_x)

        if self.input_filter == self.output_filter:
            return layers.add([inputs, res_x])
        else:
            identity = self.ConV_connect(inputs)
            assert identity.shape[-1] == self.output_filter
            return layers.add([identity, res_x])


# test the module
if __name__ == '__main__':
    x = tf.ones(shape=(2, 5, 5, 100, 100, 3))
    model = ReNet3D18L(400)
    y = model(x)
    print(y.shape)