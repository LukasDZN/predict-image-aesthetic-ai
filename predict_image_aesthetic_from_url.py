import tensorflow_hub as hub
import tensorflow as tf
import requests

# Load the model once, ideally at the start of your program
model = hub.load('https://tfhub.dev/google/vila/image/1')
predict_fn = model.signatures['serving_default']

def predict_image_aesthetic_from_url(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    
    if response.status_code == 200:
        # Read the image data
        image_bytes = response.content
        
        # Perform prediction
        predictions = predict_fn(tf.constant(image_bytes))
        aesthetic_score = predictions['predictions']
        
        return aesthetic_score[0][0]
    else:
        return "Error: Unable to fetch image from URL"

image_url = 'https://www.ratepunk.com/_next/image?url=https%3A%2F%2Fi.travelapi.com%2Flodging%2F14000000%2F13380000%2F13375900%2F13375806%2F7355a10b_z.jpg&w=640&q=75'
    
print(predict_image_aesthetic_from_url(image_url))
