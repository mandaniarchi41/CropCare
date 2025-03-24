from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('secondpage.html')

# Load the trained model
model = tf.keras.models.load_model("mobilenet_model.h5")

# Define image preprocessing function (adjust based on your model's requirements)
def preprocess_image(image):
    # Resize image to match MobileNet input size (224x224 by default)
    img = image.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    # Expand dimensions to match model input (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define class labels (replace with your actual disease classes)
class_labels = [
    "Tomato Late Blight", "Powdery Mildew", "Bacterial Leaf Spot", 
    "Downy Mildew", "Aphid Infestation", "Spider Mite Damage", 
    "Early Blight", "Root Rot", "Healthy Plant"
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get the uploaded image
    file = request.files['image']
    try:
        # Open and preprocess the image
        image = Image.open(file).convert('RGB')  # Convert to RGB if needed
        input_data = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        
        # Get the predicted class label
        predicted_class = class_labels[predicted_class_idx]
        
        # Return the prediction as JSON
        return jsonify({
            'disease': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
