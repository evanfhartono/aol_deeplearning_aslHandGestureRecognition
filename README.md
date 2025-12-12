# ASL HAND GESTURE RECOGNITION
kaggle dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
---
Onedrive link folder: https://binusianorg-my.sharepoint.com/personal/evan_hartono_binus_ac_id/_layouts/15/guestaccess.aspx?share=IgC%2D4JEGzIkXSYbRbU2Pt%5F3CAWObg%5Fu4u64%5F92St1QWb%5Fq0

demo video: https://binusianorg-my.sharepoint.com/personal/evan_hartono_binus_ac_id/_layouts/15/guestaccess.aspx?share=IQB3MIAXYHMRSI-ZjgY4WA5mAZ0iWkqReYfcmynhXsTTAFg&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rx3JNw

presentation video: https://binusianorg-my.sharepoint.com/personal/evan_hartono_binus_ac_id/_layouts/15/guestaccess.aspx?share=IQBrjSvLmMOLS7JqmpykQGgKAVYAyhkWZJo4XowXNDKfnaQ&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=QCX9Yi

## 🌟 Features
- 📷 **real time asl hand gesture recognition

---

## Warning!!
if failed to install dependencies from requirements.txt, the necessaries library can be importted from Ondedrive link folder, where there is a zip file (venv_.zip). venv\aol_fastapi\Lib\site-packages\ can be copied to current venv.

## 📂 Project Structure
```
waqu-water-quality-app/              #
│  
├── app/
│   ├── main.py                      # main app, to run app using fastapi
│   ├── custom_gesture_model.pkl     # trained model for hand gesture prediction
│   ├── requirements.txt             # list of all necessaries library
│   └── run.bat                      # batch of command line instruction to run the app in one go
├── train/                            
│   ├── extract_handlandmarks.py     # handlandmarks extraction coding
│   ├── training_aolDL.ipynb         # compilation of preprocessing, training, evaluation coding
│   └── asl_alphabet_dataset.csv     # extracted handlandmarks from image dataset
├── report/   
│   ├── presentation.pdf             # project report in slide presentation
│   └── final_report.pdf             # the project full report
├── README.md                        # summary of the project and necessaries link
└── .gitignore                       # ignore file/folder for git operation
```

---

## 🛠️ Tech Stack
- **Mediapipe
- **FastAPI
- **Tensorflow
- **OpenCV

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
[git clone https://github.com/evanfhartono/######](https://github.com/evanfhartono/aol_deeplearning_aslHandGestureRecognition.git
```

### 2. Install dependencies
```bash
python -m venv venv\aol_fastapi
pip install -r requirements.txt
```

### 3. Start the #######
```bash
uvicorn main:app --reload
```

You can run it on:
- 🌐 **Web Preview**  

---

## 📸 Prototype Preview

<img width="1093" height="928" alt="5bRCOJNxks" src="" />

---

## 📚 References
- kinivi/hand-gesture-recognition-mediapipe: This is a sample program that recognizes hand signs and finger gestures with a simple MLP using the detected key points. Handpose is estimated using MediaPipe.” Accessed: Dec. 12, 2025. [Online]. Available: https://github.com/kinivi/hand-gesture-recognition-

---
