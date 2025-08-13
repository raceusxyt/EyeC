#!/bin/bash

# Set environment variables to fix macOS TensorFlow issues
export KMP_DUPLICATE_LIB_OK=True
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

# Run the Streamlit app
streamlit run eye_disease_detector.py --server.port 8501 --server.address localhost