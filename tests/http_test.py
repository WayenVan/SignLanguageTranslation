import http
import tensorflow as tf
import tensorflow.keras as keras
import json
import pickle



from http.server import BaseHTTPRequestHandler, HTTPServer

class MyHandler(BaseHTTPRequestHandler):
    def __init__(self, model):
        super(MyHandler, self).__init__()
        self._model = model;


    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = pickle.loads(data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"hello")





server_address = ("0.0.0.0", 8000)
httpd = HTTPServer(server_address, MyHandler)
httpd.serve_forever()



