from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)

import pickle

with open('lung_cancer_model4.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        if image_file:
            # Convert FileStorage to io.BytesIO
            image_data = image_file.read()
            image_data = io.BytesIO(image_data)
            
            # Load and preprocess the image
            img = image.load_img(image_data, target_size=(446, 446))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Make a prediction
            prediction = model.predict(img)

            # Assuming binary classification
            class_label = "Class 1" if prediction[0][0] < 0.5 else "Class 2"

            return jsonify({'prediction': class_label})
        else:
            return jsonify({'error': 'No image file provided.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)