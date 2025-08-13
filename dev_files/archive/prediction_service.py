#!/usr/bin/env python3
"""
Standalone prediction service for eye disease detection.
This runs TensorFlow operations in isolation to avoid threading issues.
"""

import sys
import os
import json
import warnings
from pathlib import Path

# Suppress warnings and set environment variables
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(json.dumps({"error": f"Import error: {e}"}))
    sys.exit(1)


def load_model(model_path):
    """Load TensorFlow Lite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        return None


def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        return None


def predict(interpreter, image_array):
    """Make prediction using TensorFlow Lite model"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], image_array)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])

        # Apply softmax
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)

        return probabilities.tolist()
    except Exception as e:
        return None


def main():
    if len(sys.argv) != 3:
        print(
            json.dumps(
                {
                    "error": "Usage: python prediction_service.py <model_path> <image_path>"
                }
            )
        )
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Check if files exist
    if not Path(model_path).exists():
        print(json.dumps({"error": f"Model file not found: {model_path}"}))
        sys.exit(1)

    if not Path(image_path).exists():
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    # Load model
    interpreter = load_model(model_path)
    if interpreter is None:
        print(json.dumps({"error": "Failed to load TensorFlow Lite model"}))
        sys.exit(1)

    # Preprocess image
    image_array = preprocess_image(image_path)
    if image_array is None:
        print(json.dumps({"error": "Failed to preprocess image"}))
        sys.exit(1)

    # Make prediction
    probabilities = predict(interpreter, image_array)
    if probabilities is None:
        print(json.dumps({"error": "Failed to make prediction"}))
        sys.exit(1)

    # Return results
    print(json.dumps({"probabilities": probabilities}))


if __name__ == "__main__":
    main()
