from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Keras model
with open('lung_cancer_model4.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the 'image' file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        # Get the uploaded image from the request
        image_file = request.files['image']

        # Load and preprocess the image
        img = image.load_img(image_file, target_size=(446, 446))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Make a prediction using the loaded model
        prediction = model.predict(img)

        # Assuming binary classification
        class_label = "Class 1" if prediction[0][0] < 0.5 else "Class 2"

        return jsonify({'prediction': class_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
