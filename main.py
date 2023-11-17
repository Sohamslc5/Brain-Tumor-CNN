from __future__ import division, print_function
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import glob
import re
import numpy as np
import tensorflow
import cv2
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from io import BytesIO
app = Flask(__name__)

MODEL_PATH = 'model'

model = tensorflow.keras.models.load_model(MODEL_PATH)
model.make_predict_function()  

def model_predict(file_data, model):
    nparr = np.frombuffer(file_data, np.uint8)
    IMG_SIZE = 150
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE)
    X = X / 255.0
    X = X.reshape(-1, 150, 150, 1)
    y_pred = model.predict(X)

    return y_pred



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
            # basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        # print(file_path)
        if f:
            file_data = f.read()
            preds = model_predict(file_data, model)
            # preds = model_predict(file_path, model)
            predicted_index = np.argmax(preds)
            tumor_types = {
                0: "Glioma Tumor",
                1: "Meningioma Tumor",
                2: "No Tumor",
                3: "Pituitary Tumor"
            }
            predicted_tumor_type = tumor_types.get(predicted_index, "Unknown Tumor Type")
        return predicted_tumor_type
    return "Enter correct MRI Image"

if __name__ == '__main__':
    app.run(debug=True)
