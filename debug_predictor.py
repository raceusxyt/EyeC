#!/usr/bin/env python3
import sys
import os
import json
import traceback

print("DEBUG: Starting prediction script", file=sys.stderr)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

print("DEBUG: Environment variables set", file=sys.stderr)

try:
    print("DEBUG: Importing TensorFlow...", file=sys.stderr)
    import tensorflow as tf

    print("DEBUG: TensorFlow imported successfully", file=sys.stderr)

    print("DEBUG: Configuring TensorFlow...", file=sys.stderr)
    tf.config.set_visible_devices([], "GPU")
    print("DEBUG: TensorFlow configured", file=sys.stderr)

    print("DEBUG: Importing other modules...", file=sys.stderr)
    import numpy as np
    from PIL import Image

    print("DEBUG: All modules imported", file=sys.stderr)

    if len(sys.argv) != 3:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print(f"DEBUG: Model path: {model_path}", file=sys.stderr)
    print(f"DEBUG: Image path: {image_path}", file=sys.stderr)

    # Check if files exist
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model file not found: {model_path}"}))
        sys.exit(1)

    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    print("DEBUG: Files exist, loading model...", file=sys.stderr)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    print("DEBUG: Interpreter created", file=sys.stderr)

    interpreter.allocate_tensors()
    print("DEBUG: Tensors allocated", file=sys.stderr)

    # Load image
    print("DEBUG: Loading image...", file=sys.stderr)
    image = Image.open(image_path)
    print("DEBUG: Image loaded", file=sys.stderr)

    if image.mode != "RGB":
        image = image.convert("RGB")
        print("DEBUG: Image converted to RGB", file=sys.stderr)

    image = image.resize((224, 224))
    print("DEBUG: Image resized", file=sys.stderr)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print("DEBUG: Image preprocessed", file=sys.stderr)

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("DEBUG: Model details retrieved", file=sys.stderr)

    # Set input
    interpreter.set_tensor(input_details[0]["index"], img_array)
    print("DEBUG: Input tensor set", file=sys.stderr)

    # Run inference
    print("DEBUG: Running inference...", file=sys.stderr)
    interpreter.invoke()
    print("DEBUG: Inference completed", file=sys.stderr)

    # Get output
    output = interpreter.get_tensor(output_details[0]["index"])
    print("DEBUG: Output retrieved", file=sys.stderr)

    # Apply softmax
    output_flat = output.flatten()
    exp_values = np.exp(output_flat - np.max(output_flat))
    probabilities = exp_values / np.sum(exp_values)
    print("DEBUG: Softmax applied", file=sys.stderr)

    # Return results
    result = {"probabilities": probabilities.tolist()}
    print("DEBUG: Returning results", file=sys.stderr)
    print(json.dumps(result))

except Exception as e:
    print(f"DEBUG: Exception occurred: {str(e)}", file=sys.stderr)
    print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr)
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
