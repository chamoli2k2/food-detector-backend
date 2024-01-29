from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from io import BytesIO
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load model and dictionary
loaded_model = None
index_to_class = None

def load_model_and_dictionary():
    global loaded_model, index_to_class
    if loaded_model is None:
        loaded_model = tf.keras.models.load_model("final_model")
    if index_to_class is None:
        index_to_class = pickle.load(open('index_to_class.pkl', 'rb'))

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Define route for model prediction
@app.route('/', methods=['POST'])
def predict():
    # Load model and dictionary if not loaded
    load_model_and_dictionary()
    
    # Get image data from request
    image_file = request.files['image']
    
    # Load image
    img = Image.open(BytesIO(image_file.read()))
    
    # Preprocess image
    img_array = preprocess_image(img)
    
    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = index_to_class[predicted_class_index]
    
    # Return prediction as JSON response
    return jsonify({'class': predicted_class})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
