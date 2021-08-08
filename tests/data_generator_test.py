import json
import mediapipe as mp
from models.preprocessing.data_generator import DataGenerator
from models.preprocessing.vocabulary import WordVocab, GlossVocab

mp_holistic = mp.solutions.holistic
data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"
with open("data/Dataset/Data/dataSet.json") as file:
    content = file.read()
    data = json.loads(content)

word_vocab = WordVocab()
gloss_vocab = GlossVocab()

max_gloss_length = 0
max_sentence_length = 0

for item in data:
    glosses = [sign["gloss"] for sign in item["signs"]]
    if len(glosses) > max_gloss_length:
        max_gloss_length = len(glosses)

    gloss_vocab.fit_texts([glosses])
    word_vocab.fit_texts([item["sentence"]])

    sentence_length = len(word_vocab.sentences2sequences([item["sentence"]])[0])
    if sentence_length > max_sentence_length:
        max_sentence_length = sentence_length

#fit special tokens
word_vocab.fit_texts([["<BOS>", "<EOS>"]])


print(word_vocab.get_dictionary())
print(gloss_vocab.get_dictionary())
print(word_vocab.sentences2sequences(["<BOS> my favourate"]))
print(word_vocab.sequences2sentences([[1, 2, 0, 0]]))
print(max_sentence_length, max_gloss_length)

with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2) as holistic:

    generator = DataGenerator(1, data, gloss_vocab, word_vocab, 8,
                              8, data_set_path, holistic, mediate_output_dim=256, gloss_only=True)

    for i, item in enumerate(generator):
        print(item[1][0])
