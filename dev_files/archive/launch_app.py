#!/usr/bin/env python3
"""
Launcher script for the eye disease detection app.
This script handles environment setup and launches Streamlit safely.
"""

import os
import sys
import subprocess


def setup_environment():
    """Set up environment variables to prevent TensorFlow issues"""
    env_vars = {
        "KMP_DUPLICATE_LIB_OK": "True",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "PYTHONPATH": os.getcwd(),
    }

    for key, value in env_vars.items():
        os.environ[key] = value


def main():
    print("🚀 Starting Eye Disease Detection App...")
    print("📋 Setting up environment...")

    setup_environment()

    # Check if required files exist
    required_files = ["model.tflite", "labels (1).txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)

    print("✅ All required files found")
    print("🌐 Launching Streamlit app...")

    try:
        # Launch Streamlit with the no-tf version
        subprocess.run(
            [
                "streamlit",
                "run",
                "eye_detector_no_tf.py",
                "--server.port",
                "8505",
                "--server.address",
                "localhost",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")


if __name__ == "__main__":
    main()
