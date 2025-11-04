# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model at startup
model = tf.keras.models.load_model('super-duper.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from request
        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        img = cv2.resize(img, (48, 48))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img)[0]
        
        # Apply bias correction
        bias = np.array([1.3, 1.1, 0.9, 1.0, 1.5, 1.2, 0.7])
        adjusted = predictions * bias
        adjusted = adjusted / adjusted.sum()
        
        # Format response
        results = {
            'emotion': emotions[np.argmax(adjusted)],
            'confidence': float(np.max(adjusted)),
            'probabilities': {emotions[i]: float(adjusted[i]) for i in range(7)}
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)