# 👁️ Eye Disease Detection App

A Streamlit-based web application that demonstrates AI-powered eye disease detection using a TensorFlow Lite model. This app provides an intuitive interface for uploading eye images and receiving AI-powered analysis results.

## 🎯 Features

- **Interactive Web Interface**: Clean, user-friendly Streamlit interface
- **Real-time Analysis**: Upload and analyze eye images instantly
- **Multiple Disease Detection**: Detects 3 different eye conditions
- **Confidence Scoring**: Shows prediction confidence levels
- **Educational Demo**: Demonstrates AI medical screening workflow
- **Cross-platform**: Works on Windows, Android, macOS, and Linux

## 🔍 Detected Conditions

The AI model can identify the following eye conditions:

1. **⚪ Cataract** - Clouding of the eye's lens
2. **🔴 Diabetic Retinopathy** - Diabetes-related retinal damage
3. **🩸 Glaucoma** - Optic nerve damage from eye pressure
4. **✅ Normal** - Healthy eye condition

## 🚀 Quick Start

### Option 1: Using the Launcher (Recommended)

```bash
# Clone or download the project files
# Navigate to the project directory
cd EyeC

# Run the launcher script
python run.py
```

The launcher will:

- ✅ Check for required files
- 📦 Install dependencies automatically
- 🚀 Launch the app in your browser

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📁 Required Files

Make sure these files are in your project directory:

- ✅ `app.py` - Main Streamlit application
- ✅ `labels (1).txt` - Class labels for the model
- ✅ `requirements.txt` - Python dependencies
- ✅ `run.py` - Launcher script
- ✅ `model.tflite` - TensorFlow Lite model

## 💻 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 50MB for dependencies
- **Internet**: Required for initial dependency installation

### Dependencies
- `streamlit>=1.28.0` - Web interface framework
- `pillow>=9.5.0` - Image processing
- `numpy==1.24.0` - Numerical computations
- `tflite-runtime==2.14.0`
  
## 📖 How to Use

1. **Launch the App**:

   - Run `python run.py` or `streamlit run app.py`
   - Open your browser to `http://localhost:8501`

2. **Upload an Image**:

   - Click "Upload Eye Image for Analysis"
   - Select a PNG, JPG, or JPEG file
   - Ensure the image is clear and well-lit

3. **Run Analysis**:

   - Click "🚀 Run Disease Detection"
   - Wait for the AI analysis to complete

4. **View Results**:
   - See the primary detection result
   - Review confidence scores for all conditions
   - Check detailed analysis breakdown

## 📷 Image Guidelines

For best results, use images that are:

- ✅ **Clear and in focus**
- ✅ **Well-lit** (good lighting conditions)
- ✅ **High resolution** (224x224 pixels or higher)
- ✅ **Eye-centered** (eye should be the main subject)
- ❌ Avoid blurry, dark, or low-quality images

## 🔧 Technical Details

### Current Status: Demo Mode

- ✅ Demonstrates the complete user interface
- ✅ Shows realistic prediction workflows
- ✅ Provides educational value about AI medical screening
- ⚠️ Uses simulated predictions (not actual AI inference)

### Model Information

- **Format**: TensorFlow Lite (.tflite)
- **Input Size**: 224×224 RGB images
- **Output**: 4-class probability distribution
- **Architecture**: Convolutional Neural Network
- **Training**: Pre-trained on eye disease dataset

### For Production Deployment

To use with actual AI inference:

1. **Linux Environment**: Deploy on Linux servers for full TensorFlow support
2. **Docker**: Use containerized environments for consistency
3. **Cloud APIs**: Consider cloud-based inference services

## 🛠️ Troubleshooting

### Common Issues

**App won't start:**

```bash
# Update pip and reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Port already in use:**

```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**Missing files error:**

- Ensure all required files are in the project directory
- Check file names match exactly (case-sensitive)

**Browser doesn't open:**

- Manually navigate to `http://localhost:8501`
- Try a different browser
- Check firewall settings

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for **educational and demonstration purposes only**.

- 🚫 **NOT for medical diagnosis**
- 🚫 **NOT a substitute for professional medical advice**
- 🚫 **NOT for treatment decisions**

**Always consult qualified healthcare professionals** for proper medical evaluation and diagnosis of eye conditions.

## 🔄 Development & Customization

### Modifying the App

- **UI Changes**: Edit `app.py` to customize the Streamlit interface
- **Model Replacement**: Replace `model.tflite` with your trained model
- **Labels Update**: Modify `labels.txt` for different classes
- **Styling**: Add custom CSS in the Streamlit app

### File Structure

```
EyeC/
├── app.py                 # Main application
├── run.py                 # Launcher script
├── requirements.txt       # Dependencies
├── labels.txt             # Model labels
├── model.tflite           # AI model
├── .streamlit
    └── config             # streamlit configaration
├── dev_files
   ├── run.py                        # Launcher script
   ├── eye_detector_demo.py          # demo app
   ├── simple_eye_detector.py        # development file
   └── archive                       # development files
└── README.md            # This file
```

## 📄 License

This project is for educational purposes. Ensure you have appropriate licenses for any models or datasets used in production environments.

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the technical documentation
3. Ensure all required files are present
4. Verify Python and dependency versions

---

**Made with ❤️ for educational AI and healthcare demonstration**


