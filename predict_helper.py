
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json

def predict_image(model_path, image_path):
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Load and preprocess image
        from PIL import Image
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)
        
        return probabilities.tolist()
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    result = predict_image(model_path, image_path)
    print(json.dumps(result))
