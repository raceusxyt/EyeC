
import sys
import os
import json

# Minimal environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    
    # Disable all optimizations that might cause threading issues
    tf.config.optimizer.set_jit(False)
    tf.config.set_visible_devices([], 'GPU')
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Process image
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    
    # Predict
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index']).flatten()
    
    # Manual softmax
    exp_output = np.exp(output - np.max(output))
    probabilities = exp_output / np.sum(exp_output)
    
    print(json.dumps({"probabilities": probabilities.tolist()}))
    
except Exception as e:
    print(json.dumps({"error": str(e)}))
