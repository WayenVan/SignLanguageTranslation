"""Http Server"""
import sys
import os

sys.path.append(os.getcwd())

from models.preprocessing.vocabulary import WordVocab, GlossVocab
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from queue import Queue
import gtts
from playsound import playsound
import pickle

class MyHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        s = pickle.loads(data)
        q.put(word_vocab.sequences2sentences([s[:-1]])[0])
        print("put done")
        #response
        self.send_response(200)
        self.end_headers()

def speaking(que: Queue):
    while True:
        string = que.get(block=True)
        tts = gtts.gTTS(string)
        tts.save("Demo/template.mp3")
        playsound("Demo/template.mp3")


"""load data"""
word_vocab = WordVocab()
gloss_vocab = GlossVocab()
word_vocab.load(os.getcwd() + "/data/vcab/word_vocab.json")
gloss_vocab.load(os.getcwd() + "/data/vcab/gloss_vocab.json")

"""speaking thread"""
q = Queue()
threading.Thread(target=speaking, args=(q,)).start()

"""http thread"""
server_address = ("0.0.0.0", 2333)
httpd = HTTPServer(server_address, MyHandler)
httpd.serve_forever()