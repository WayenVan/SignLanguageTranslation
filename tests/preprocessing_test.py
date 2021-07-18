import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from tensorflow.keras.utils import to_categorical
from tensorflow import data


texts = ["today is a beautiful day up to you",
            "ahah all3 in3 for_3 fuck"]

tk = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', oov_token="<OOV>")
tk.fit_on_texts(texts=texts)

s = tk.texts_to_sequences(texts)
s = sequence.pad_sequences(s)
s = to_categorical(s ,num_classes=len(tk.word_index) + 1)
print(s)



