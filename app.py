from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

label_encoder = LabelEncoder()

# Load the pickled model at startup
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model.pkl')
try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")

        # Extract values from the JSON in the correct order
        input_data = np.array([[
            float(data['cgpa']),
            float(data['sub1']),
            float(data['sub2']),
            float(data['sub3']),
            float(data['sub4']),
            float(data['sub5']),
            float(data['sub6']),
            float(data['sub7']),
            float(data['sub8']),
            float(data['sub9'])
        ]])
        
        logger.info(f"Processed input data shape: {input_data.shape}")

        # Get prediction from the loaded model
        prediction = model.predict(input_data)
        predicted_elective = np.argmax(prediction, axis=1)[0]  # Get single prediction
        
        logger.info(f"Prediction result: {predicted_elective}")
        
        return jsonify({
            "predicted_elective": int(predicted_elective)  # Convert to int for JSON serialization
        })
        
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)
