from flask import Flask, request, jsonify
import tensorflow_hub as hub
import tensorflow as tf
import requests

app = Flask(__name__)

# Load the TensorFlow model
model = hub.load('https://tfhub.dev/google/vila/image/1')
predict_fn = model.signatures['serving_default']

def predict_image_aesthetic_from_url(image_url):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image_bytes = response.content
        predictions = predict_fn(tf.constant(image_bytes))
        aesthetic_score = predictions['predictions']
        return aesthetic_score.numpy().tolist()
    else:
        return "Error: Unable to fetch image from URL"

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    score = predict_image_aesthetic_from_url(image_url)
    return jsonify({'aesthetic_score': score})

if __name__ == '__main__':
    app.run(debug=True)
