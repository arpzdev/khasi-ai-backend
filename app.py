from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import numpy as np
import traceback
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads/'
MODEL_PATH = 'model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16 MB

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Define a function to check if the image resembles red meat
def resembles_red_meat(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Check the proportion of red pixels in the image
        red_channel = img_array[:, :, 0]  # Red channel
        green_channel = img_array[:, :, 1]  # Green channel
        blue_channel = img_array[:, :, 2]  # Blue channel
        
        # Define thresholds for red pixel detection (more lenient)
        red_threshold = 130  # Lowered minimum value for red
        green_threshold = 120  # Increased maximum value for green
        blue_threshold = 120  # Increased maximum value for blue

        # Count the number of red-like pixels
        red_pixels = np.sum((red_channel > red_threshold) & 
                            (green_channel < green_threshold) & 
                            (blue_channel < blue_threshold))

        # Calculate the total number of pixels
        total_pixels = img_array.shape[0] * img_array.shape[1]

        # Determine if the image resembles red meat (e.g., red pixels should be more than 10% of total pixels)
        if red_pixels / total_pixels > 0.10:  # Adjusted to 10%
            return True
        else:
            return False
    except Exception as e:
        print("Error in red meat check:", e)
        return False  # Default to False if there's an error

# Define the prediction function
def predict_image(image_path):
    if not resembles_red_meat(image_path):
        return 'meat image not recognized'

    try:
        # Load and preprocess the image
        img = Image.open(image_path).resize((416, 416))
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        classes = ['Fresh', 'Half-Fresh', 'Spoiled']
        predicted_class = classes[np.argmax(predictions)]

        return predicted_class
    except Exception as e:
        print("Error during prediction:", e)
        raise

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Log the filename for debugging
        print("Received file:", file.filename)

        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print("Saving file to:", image_path)  # Log where the file will be saved
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((416, 416))
        image.save(image_path)

        # Predict the freshness of the uploaded image
        prediction = predict_image(image_path)
        print("Prediction:", prediction)  # Log the prediction result

        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename, 'prediction': prediction}), 200

    except Exception as e:
        error_message = str(e)  # Capture the error message
        print("Error occurred during file upload:", error_message)  # Log the error
        return jsonify({'error': error_message, 'trace': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))  # Added host and port
  # Run the app in debug mode