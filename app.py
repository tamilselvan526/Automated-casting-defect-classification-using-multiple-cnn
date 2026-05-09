import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.preprocessing import image
from pathlib import Path
import uuid
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app) # Enable CORS to allow root index.html to work from port 5500

# Configuration
MODEL_DIR = Path("model")
METADATA_PATH = MODEL_DIR / "metadata.json"
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load Metadata and Models once at startup
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

metadata = load_metadata()
loaded_models = {}

if metadata:
    print("Loading models into memory...")
    for model_name, path_str in metadata["models"].items():
        model_path = Path(path_str.replace('\\', '/'))
        if model_path.exists():
            try:
                print(f" - Loading {model_name}...")
                loaded_models[model_name] = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"Warning: model file not found for {model_name}: {model_path}")

def load_and_preprocess_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_result(predicted_label):
    if predicted_label.lower() == "normal":
        return "Non-Defective", None
    return "Defective", predicted_label

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    # Save temp file
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    if not metadata:
        return jsonify({"error": "Model metadata not found. Please train models first."}), 500

    img_size = metadata.get("img_size", 224)
    class_names = metadata.get("class_names", [])

    try:
        img_array = load_and_preprocess_image(filepath, img_size)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    results = []
    # Use cached models for instant results
    model_paths = metadata["models"]
    for model_name in model_paths.keys():
        if model_name in loaded_models:
            try:
                model = loaded_models[model_name]
                prediction = model.predict(img_array, verbose=0)[0]
                predicted_index = int(np.argmax(prediction))
                predicted_label = class_names[predicted_index].capitalize()
                confidence = float(prediction[predicted_index]) * 100
                status, _ = format_result(predicted_label)

                results.append({
                    "model": model_name,
                    "label": predicted_label,
                    "status": status,
                    "confidence": f"{confidence:.2f}%",
                    "confidence_val": confidence
                })
            except Exception as e:
                results.append({
                    "model": model_name,
                    "label": "Analysis Error",
                    "status": str(e),
                    "confidence": "0.00%",
                    "confidence_val": 0
                })
        else:
            results.append({
                "model": model_name,
                "label": "N/A",
                "status": "Model not loaded",
                "confidence": "0.00%",
                "confidence_val": 0
            })

    return jsonify({
        "results": results,
        "filename": filename
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Optimized AI Casting Guard Server...")
    app.run(debug=True, port=5000)
