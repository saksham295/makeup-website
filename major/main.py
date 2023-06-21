# -*- coding: utf-8 -*-
from flask import Flask, send_file, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from imageio import imread, imsave
import os
import cv2

app = Flask(__name__)
CORS(app)
img_size = 256

class DMT(object):
    def __init__(self):
        self.pb = 'dmt.pb'
        self.style_dim = 8

    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2

    def load_image(self, path):
        img = cv2.resize(imread(path), (img_size, img_size))
        img_ = np.expand_dims(self.preprocess(img), 0)
        return img / 255., img_

    def load_model(self):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()

            with open(self.pb, 'rb') as fr:
                output_graph_def.ParseFromString(fr.read())
                tf.import_graph_def(output_graph_def, name='')

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            graph = tf.get_default_graph()
            self.X = graph.get_tensor_by_name('X:0')
            self.Y = graph.get_tensor_by_name('Y:0')
            self.S = graph.get_tensor_by_name('S:0')
            self.X_content = graph.get_tensor_by_name('content_encoder/content_code:0')
            self.X_style = graph.get_tensor_by_name('style_encoder/style_code:0')
            self.Xs = graph.get_tensor_by_name('decoder_1/g:0')
            self.Xf = graph.get_tensor_by_name('decoder_2/g:0')

    def pairwise(self, A, B):
        A_img, A_img_ = self.load_image(A)
        B_img, B_img_ = self.load_image(B)
        Xs_ = self.sess.run(self.Xs, feed_dict={self.X: A_img_, self.Y: B_img_})

        result = np.ones((img_size, 3 * img_size, 3))
        result[:, :img_size] = A_img
        result[:, img_size: 2 * img_size] = B_img
        result[:, 2 * img_size:] = self.deprocess(Xs_)[0]
        imsave(os.path.join('output', 'pairwise.jpg'), result)

    

model = DMT()
model.load_model()

@app.route('/pairwise', methods=['POST'])
@cross_origin()
def pairwiseApi():
    print(request.files)
    no_makeup_file = request.files['no_makeup']
    makeup_file = request.files['makeup']

    no_makeup_path = 'faces/no_makeup/no_makeup.jpg'
    makeup_path = 'faces/makeup/makeup.jpg'
    no_makeup_file.save(no_makeup_path)
    makeup_file.save(makeup_path)

    model.pairwise(no_makeup_path, makeup_path)
    generated_image_path = 'output/pairwise.jpg'

    return send_file(generated_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max upload size (e.g., 16MB)
    app.run(host='127.0.0.1', port=7001)