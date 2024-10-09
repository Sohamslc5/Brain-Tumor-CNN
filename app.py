from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load your model (change path if needed)
model = load_model('saved_model/my_model.keras')

def preprocess_input_image(img_path):
    # Load the image with target size of (150, 150)
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")  # Resize to (150, 150)
    # Convert the image to array
    img = image.img_to_array(img)
    print(img.shape)
    test = [img]
    test = np.array(test)
    
    return test

def model_predict(img_path, model):
    img = preprocess_input_image(img_path)
    preds = model.predict(img)
    print(preds)
    return preds

# Route for homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Make prediction
        preds = model_predict(filepath, model)
        
        # Convert the prediction to a meaningful label
        result = np.argmax(preds, axis=1)
        labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        prediction = labels[result[0]]

        return render_template('result.html', prediction=prediction, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
