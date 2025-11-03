"""
CatDog Vision AI - Flask Backend API
Provides /predict endpoint for image classification
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend

# Load model (lazy loading)
model = None
MODEL_PATH = "catdog_model.h5"
IMG_SIZE = 150

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                "error": "Model not found. Please train the model first."
            }), 404
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    return model

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict endpoint for image classification"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Load model
        model = load_model()
        if isinstance(model, tuple):  # Error response
            return model
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Interpret result
        # Model outputs: 0 = Cat, 1 = Dog (for binary classification)
        if confidence < 0.5:
            predicted_class = "Cat"
            confidence_score = 1 - confidence
        else:
            predicted_class = "Dog"
            confidence_score = confidence
        
        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence_score, 4),
            "raw_confidence": round(confidence, 4)
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during prediction"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/api', methods=['GET'])
def api_info():
    """API info endpoint"""
    return jsonify({
        "status": "CatDog Vision AI API is running",
        "endpoints": {
            "/api/predict": "POST - Upload image for classification",
            "/api/health": "GET - Health check"
        }
    })

if __name__ == '__main__':
    print("Starting CatDog Vision AI API...")
    print(f"Model path: {MODEL_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)

