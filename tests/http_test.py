import http
from http.server import BaseHTTPRequestHandler, HTTPServer

def run(server_class, handler_class):
    server_address = ("192.168.8.183", 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

run(HTTPServer, BaseHTTPRequestHandler)

