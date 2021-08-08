import tensorflow as tf
import tensorflow.keras as keras
from models.transformer.position_embedding import PositionEmbedding

lay = PositionEmbedding(256)

x = keras.Input(shape=(None, 256))
x = lay(x)
print(x)