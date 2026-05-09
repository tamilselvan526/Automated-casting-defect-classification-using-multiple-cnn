import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import utils as keras_utils


MODEL_DIR = Path("model")
METADATA_PATH = MODEL_DIR / "metadata.json"
DEFAULT_IMAGE = "dataset/test/blowhole/cast_def_0_5305.jpeg"
FALLBACK_MODELS = {
    "Custom CNN": [MODEL_DIR / "custom_cnn.keras", MODEL_DIR / "casting_model.keras", MODEL_DIR / "casting_model.h5"],
    "MobileNet": [MODEL_DIR / "mobilenet.keras"],
    "ResNet": [MODEL_DIR / "resnet.keras"],
}


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_image(img_path, img_size):
    img = keras_utils.load_img(img_path, target_size=(img_size, img_size))
    img_array = keras_utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def format_result(predicted_label):
    if predicted_label.lower() == "normal":
        return "Not Defective", None
    return "Defective", predicted_label


def resolve_model_path(model_name, configured_path):
    candidate_paths = [Path(configured_path), *FALLBACK_MODELS.get(model_name, [])]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def print_missing_model_message(model_name, configured_path):
    print("=" * 60)
    print(f"{model_name} Output")
    print("=" * 60)
    print("Status: Model file not found")
    print(f"Expected Path: {configured_path}")
    print("Message: Train this model first by running main.py")


def predict_with_model(model_name, model_path, img_array, class_names):
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index].capitalize()
    confidence = float(prediction[predicted_index]) * 100
    status, _ = format_result(predicted_label)

    return {
        "model": model_name,
        "label": predicted_label,
        "status": status,
        "confidence": f"{confidence:.2f}%",
    }


def main():
    metadata = load_metadata()
    img_size = metadata["img_size"]
    class_names = metadata["class_names"]
    model_paths = metadata["models"]

    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    
    if not os.path.exists(img_path):
        print(f"Error: The image file '{img_path}' was not found.")
        return

    img_array = load_image(img_path, img_size)

    print("=" * 60)
    print("Casting Defect Classification")
    print("=" * 60)
    print(f"Input Image: {img_path}")

    results = []

    for model_name, configured_path in model_paths.items():
        resolved_model_path = resolve_model_path(model_name, configured_path)
        if resolved_model_path is None:
            print_missing_model_message(model_name, configured_path)
            continue

        res = predict_with_model(model_name, resolved_model_path, img_array, class_names)
        results.append(res)

    if results:
        print("=" * 68)
        print(f"{'Model Name':<15} | {'Classification':<15} | {'Status':<15} | {'Confidence':<10}")
        print("-" * 68)
        for r in results:
            print(f"{r['model']:<15} | {r['label']:<15} | {r['status']:<15} | {r['confidence']:<10}")
    else:
        print("=" * 68)
        print("No trained model files are available yet. Run main.py first.")
    print("=" * 68)


if __name__ == "__main__":
    main()
