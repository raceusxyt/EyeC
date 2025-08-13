#!/usr/bin/env python3
"""
Eye Disease Detection App Launcher
Simple launcher script for the Eye Disease Detection Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required files exist"""
    required_files = ["app.py", "labels.txt", "requirements.txt"]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # Check optional model file
    if not Path("model.tflite").exists():
        print("⚠️  Model file 'model.tflite' not found - app will run in demo mode")
    else:
        print("✅ Model file detected")

    return True


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def launch_app():
    """Launch the Streamlit app"""
    print("🚀 Launching Eye Disease Detection App...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop the app")
    print("=" * 50 + "\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")


def main():
    """Main launcher function"""
    print("👁️  Eye Disease Detection App Launcher")
    print("=" * 40)

    # Check if we're in the right directory
    if not check_requirements():
        print("\n💡 Make sure you're running this script from the project directory")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Launch the app
    launch_app()


if __name__ == "__main__":
    main()

