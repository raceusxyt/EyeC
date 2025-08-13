
import sys
import os
import json
import warnings

# Completely isolate TensorFlow
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Use fork to completely isolate the process
if hasattr(os, 'fork'):
    pid = os.fork()
    if pid == 0:  # Child process
        try:
            import tensorflow as tf
            import numpy as np
            from PIL import Image
            
            def run_prediction(model_path, image_path):
                # Load model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                # Load and preprocess image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize((224, 224))
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Softmax
                exp_output = np.exp(output[0] - np.max(output[0]))
                probabilities = exp_output / np.sum(exp_output)
                
                return probabilities.tolist()
            
            model_path = sys.argv[1]
            image_path = sys.argv[2]
            result = run_prediction(model_path, image_path)
            print(json.dumps({"probabilities": result}))
            
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        
        os._exit(0)  # Exit child process
    else:
        # Parent process waits for child
        os.waitpid(pid, 0)
else:
    # Fallback for systems without fork
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)
        
        print(json.dumps({"probabilities": probabilities.tolist()}))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
