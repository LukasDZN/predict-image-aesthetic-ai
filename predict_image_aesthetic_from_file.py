import tensorflow_hub as hub
import tensorflow as tf

# Load the model once, ideally at the start of your program
model = hub.load('https://tfhub.dev/google/vila/image/1')
predict_fn = model.signatures['serving_default']

def predict_image_aesthetic(image_path):
    # Load image bytes
    image_bytes = open(image_path, 'rb').read()
    
    # Perform prediction
    predictions = predict_fn(tf.constant(image_bytes))
    aesthetic_score = predictions['predictions']
    
    return aesthetic_score

image_path_list = [
    'pr_1.jpg',
    'pretty_hotel.jpg',
    'ugly_hotel.jpg',
    'church_inside.jpg',
    'church_outside_bright.jpg',
    'church_outside_wide.jpeg'
]

print('✅ SCORES:')

for image_path in image_path_list:
    aesthetic_score = predict_image_aesthetic(image_path)
    print(f'path: "{image_path}" - score: {aesthetic_score[0][0]}')
