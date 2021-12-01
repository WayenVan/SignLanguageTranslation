import json
import os
import pickle
import sys
import requests

sys.path.append(os.getcwd())

from http.server import BaseHTTPRequestHandler, HTTPServer
from models.sign_translation import create_sign_translation_model
from models.preprocessing.vocabulary import WordVocab, GlossVocab

response_server_adress = "http://192.168.8.183:2333"

video_input_shape=None #get from dataset
word_input_shape=None #get from dataset for max sentence sequence
video_embed_dim=256
word_embed_dim=256
gloss_categories=39 #get from dataset
word_categories=46 #get from dataset
num_block_encoder=6
num_block_decoder=12
head_num=12
k_dim=64
v_dim=64
ff_dim=2048
encoder_linear_hidden_dim=256
decoder_linear_hidden_dim=256
drop_out=0.1

"""prepare data"""
data_set_path = "/home/wayenvan/SignLanguageTranslation/data/DataSet/Data"
# data_set_path = "/Users/wayenvan/Desktop/MscProject/TempPro/data/DataSet/Data"
with open(data_set_path + "/dataSet.json") as file:
    content = file.read()
    data = json.loads(content)

data_length = len(data)
word_vocab = WordVocab()
gloss_vocab = GlossVocab()
word_vocab.load(os.getcwd() + "/data/vcab/word_vocab.json")
gloss_vocab.load(os.getcwd() + "/data/vcab/gloss_vocab.json")

# calculate max length
max_gloss_length = 0
max_sentence_length = 0

for item in data:
    glosses = [sign["gloss"] for sign in item["signs"]]
    if len(glosses) > max_gloss_length:
        max_gloss_length = len(glosses)

    sentence_length = len(word_vocab.sentences2sequences([item["sentence"]])[0])
    if sentence_length > max_sentence_length:
        max_sentence_length = sentence_length

print("data length: ", len(data))
print(gloss_vocab.get_dictionary()[0])
print(word_vocab.get_dictionary()[0])

assert gloss_categories == len(gloss_vocab.get_dictionary()[0])
assert word_categories == len(word_vocab.get_dictionary()[0])
word_input_shape = max_sentence_length
video_input_shape = (max_gloss_length, 5, 256, 256, 3)

model = create_sign_translation_model(video_input_shape=video_input_shape,
                                      word_input_shape=word_input_shape,
                                      video_embed_dim=video_embed_dim,
                                      word_embed_dim=word_embed_dim,
                                      gloss_categories=gloss_categories,
                                      word_categories=word_categories,
                                      num_block_encoder=num_block_encoder,
                                      num_block_decoder=num_block_decoder,
                                      head_num=head_num,
                                      k_dim=k_dim,
                                      v_dim=v_dim,
                                      ff_dim=ff_dim,
                                      encoder_linear_hidden_dim=encoder_linear_hidden_dim,
                                      decoder_linear_hidden_dim=decoder_linear_hidden_dim,
                                      drop_out=drop_out)
#load model params
model.load_weights(os.getcwd() + "/data/training_data/checkpoint/checkpoint")
model.trainable=False
model.summary()

bos_index = word_vocab.tokenizer.word_index["<bos>"]
eos_index = word_vocab.tokenizer.word_index["<eos>"]

"""Http Server"""
class MyHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = pickle.loads(data)
        ret = model.iterative_prediction(data, bos_index=bos_index, eos_index=eos_index)

        #response
        self.send_response(200)
        self.end_headers()
        #self.wfile.write(pickle.dumps(ret))
        requests.post(response_server_adress, data=pickle.dumps(ret))

        #send result to response_server

server_address = ("0.0.0.0", 2333)
httpd = HTTPServer(server_address, MyHandler)
httpd.serve_forever()



