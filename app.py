import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from PIL import Image
from io import BytesIO
import pickle


# Initialize Flask app
app = Flask(__name__)
CORS(app)

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(80, activation='softmax')(x) 

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights("model_weights.h5")

    return model


# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Model and Dictionary
model = None
index_to_class = None

# Load dictionary 
if index_to_class is None:
    index_to_class = pickle.load(open('index_to_class.pkl', 'rb'))

# Create model
model=create_model(80)

# Define route for model prediction
@app.route('/', methods=['GET','POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'class': 'image not found'}) , 400
    
    # Get image data from request
    image_file = request.files['image']

    allowed_extensions = {'jpg', 'jpeg', 'png'} 
    if image_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'class': 'invalid image format'}) , 400
    
    # Load image
    img = Image.open(BytesIO(image_file.read()))
    
    # Preprocess imageEncoder
    img_array = preprocess_image(img)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = index_to_class[predicted_class_index]
    
    # Return prediction as JSON response
    return jsonify({'class':predicted_class})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
