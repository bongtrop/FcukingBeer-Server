# -*- coding: utf-8 -*-

'''
Author: Pongsakorn Sommalai (bongtrop@gmail.com)
Date: 25/11/2015
Description: Server blur beer photo
'''

# Default
import json
import cgi
import random
import time
import detect

# Web Server
from twisted.web.server import Site
from twisted.web.resource import Resource
from twisted.internet import reactor

def gen_random_name(l):
   return ''.join([random.choice('0123456789ABCDEF') for x in range(l)])

class Blur(Resource):
    isLeaf = True

    def render_GET(self, request):
        try:
            f = open("out.jpg", "rb")
            request.setHeader("content-type", "image/jpeg")
            return f.read()
        except Exception as error:
            return "No Init"

    def render_POST(self, request):
        #try:
            self.headers = request.getAllHeaders()

            img = cgi.FieldStorage(
                fp = request.content,
                headers = self.headers,
                environ = {'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['content-type']})

            filename = gen_random_name(32)+'.jpg'

            out = open('images/'+filename, 'wb')
            out.write(img["image"].value)
            out.close()

            rects = detect.beer('images/'+filename)

            return json.dumps({"status": "success", "data": rects})
        #except Exception as error:
        #    request.setResponseCode(400)
        #    return json.dumps({"status": "fail", "data": str(error)})

resource = Blur()
factory = Site(resource)
reactor.listenTCP(12345, factory)
reactor.run()
