import requests

# Load an image for testing
image_path = 'test2.png'

# Send a POST request to the Flask app for prediction
response = requests.post('http://127.0.0.1:5000/predict', files={'image': open(image_path, 'rb')})

if response.status_code == 200:
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print("Error:", response.text)
