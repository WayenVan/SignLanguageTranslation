import sys
import os
import pickle
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from models import losses as self_defined_losses
import models

from models.video2gloss import create_video2gloss_model

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, newshape=(60000, 28, 28, 1))
    x_train = tf.image.per_image_standardization(x_train)
    x_train = tf.reshape(x_train, shape=(10000, 6, 4, 7, 28, 1))

    y_train = to_categorical(y_train, num_classes=10)
    y_train = tf.reshape(y_train, shape=(10000, 6, 10))
    blank = tf.zeros(shape=(10000, 6, 128))


    #create model
    model = create_video2gloss_model(input_shape=(6, 4, 7, 28, 1),
                                     video_embed_dim=128,
                                     block_number=6,
                                     encoder_head_number=6,
                                     ff_dim=64,
                                     linear_hidden_dim=256,
                                     linear_output_dim=10,
                                     drop_out=0.1)

    opt = optimizers.SGD()

    model.summary()
    model.compile(optimizer=opt, loss=[self_defined_losses.cross_entrophy,
                                        self_defined_losses.cross_entrophy],
                  loss_weights=[1, 0])
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.getcwd()+"/data/checkpoint",
        save_weights_only=True,
    )
    model.load_weights(os.getcwd()+"/data/checkpoint")

    history = model.fit(x=x_train[:500], y=[y_train[:500], blank[:500]],
                        batch_size=4,
                        epochs=50,
                        verbose=1,
                        callbacks=[checkpoint_callback])

    with open(os.getcwd()+"/data/history", "w+") as file:
        pickle.dump(history, file)
