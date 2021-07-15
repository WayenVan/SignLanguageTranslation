import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
from models.cnn.resnet3D import ReNet3D18L
# test the module
if __name__ == '__main__':
    x = tf.ones(shape=(4, 5, 5, 50, 50, 3))
    model = ReNet3D18L(128)
    y = model(x)
    print(y.shape)