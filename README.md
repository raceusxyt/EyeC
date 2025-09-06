## 👁️ EyeC – AI-Powered Eye Disease Detection



A Streamlit-based web app that uses a TensorFlow Lite model to demonstrate AI-powered eye disease detection.
EyeC provides a simple, educational interface for uploading fundus eye images and getting AI-driven analysis.


---

## 🎯 Features

🌐 Interactive Web Interface – Clean, user-friendly Streamlit UI

⚡ Real-time Analysis – Upload and analyze eye images instantly

🧠 Multi-condition Detection – Detects 3 common eye diseases + normal

📊 Confidence Scoring – View prediction confidence for all classes

🎓 Educational Demo – Demonstrates AI-powered medical screening workflow

💻 Cross-Platform – Works on Windows, Linux, macOS, and Android (browser-based)



---

## 🔍 Detected Conditions

EyeC can classify images into the following conditions:

1. ⚪ Cataract – Clouding of the eye’s lens


2. 🔴 Diabetic Retinopathy – Retina damage caused by diabetes


3. 🩸 Glaucoma – Optic nerve damage due to high eye pressure


4. ✅ Normal – Healthy eye condition




---

## 🚀 Quick Start

### Option 1 – Using the Launcher (Recommended)  

```bash
# Clone project
git clone https://github.com/your-repo/EyeC.git
cd EyeC

# Run the launcher
python run.py
```
The launcher will:
✔️ Check for required files  
✔️ Install dependencies  
✔️ Launch the app in your browser  


---

Option 2 – Manual Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

EyeC/
├── app.py              # Main Streamlit   application  
├── run.py              # Launcher script  
├── model.tflite        # AI model   (TensorFlow Lite)  
├── labels.txt          # Class labels  
├── requirements.txt    # Dependencies  
├── .streamlit/         # Streamlit config  
├── dev_files/          # Development/demo   files  
└── README.md           # Documentation  


---

## 📷 Image Guidelines

For accurate predictions, upload fundus eye images that are:

✅ Clear and well-focused  
✅ Well-lit  
✅ High resolution (≥224×224 pixels)  
❌ Avoid blurry, dark, or off-angle images  


---

##🔧 Technical Details

Framework: Streamlit

Model: TensorFlow Lite CNN (4-class classification)

Input Size: 224×224 RGB images

Output: Probability scores for 4 conditions

Deployment: Hugging Face Spaces (demo hosting)



---

## ⚠️ Disclaimer

This project is for educational and demonstration purposes only.

🚫 Not for real medical diagnosis  
🚫 Not a replacement for professional consultation  
✅ Always consult a qualified ophthalmologist for medical advice


---

## 🤝 Contributing

Want to improve EyeC?

1. Fork the repo


2. Create a feature branch


3. Commit changes


4. Open a pull request 🚀




---

## 📞 Support

Check the Troubleshooting section in this README

Verify dependencies with pip install -r requirements.txt

For deployment issues, ensure model + labels are correctly placed



---

### 👁️ EyeC – Early Detection, Healthy Vision
Made with ❤️ for AI & healthcare education