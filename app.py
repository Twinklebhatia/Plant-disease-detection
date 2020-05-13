# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:09:47 2020

@author: Twinkle
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'G:\\deloitte technoutsav\\ML MODEL\\models\\model.h5'

# Load your trained model
from keras.models import load_model
from keras.models import model_from_json
import json

json_file = open('G:\\deloitte technoutsav\\ML MODEL\\models\\model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("G:\\deloitte technoutsav\\ML MODEL\\models\\model.h5")
print("Loaded Model from disk")
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, loaded_model):
    img = Image.open(image_path)
    size = (64, 64)
    img = img.resize(size, PIL.Image.ANTIALIAS)  
    img_array = np.array(img)
    img_array = img_array/255
    img_array=img_array.reshape(1,64,64,3)
    #plt.imshow(img)
    y=loaded_model.predict(img_array)
    if y[0][0]>=0.5:
        return "Healthy pepper belly plant"
    else:
        return "Unhealthy pepper belly plant"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, loaded_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)