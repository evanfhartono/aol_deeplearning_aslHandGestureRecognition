import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from datetime import datetime
import threading
import time
import queue

app = FastAPI()

video_camera = cv2.VideoCapture(0)
prediction_camera = cv2.VideoCapture(0)  

latest_prediction = {
    "gesture": "Waiting...",
    "confidence": 0.0,
    "top_predictions": [],
    "timestamp": None
}

class CustomGestureTester:
    def __init__(self, model_path='custom_gesture_model.pkl'):
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.label_encoder = self.model_data['label_encoder']
        self.scaler = self.model_data['scaler']
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.6
        
        print(f"Custom MLP Model loaded!")
        print(f"Recognizable gestures: {list(self.label_encoder.classes_)}")
    
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    
    def predict_gesture(self, landmarks):
        landmarks_scaled = self.scaler.transform(landmarks.reshape(1, -1))
        
        prediction_proba = self.model.predict(landmarks_scaled, verbose=0)
        prediction = np.argmax(prediction_proba, axis=1)
        confidence = np.max(prediction_proba)
        gesture = self.label_encoder.inverse_transform(prediction)[0]
        
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        top_predictions = []
        for idx in top_3_indices:
            top_predictions.append({
                'gesture': self.label_encoder.classes_[idx],
                'confidence': float(prediction_proba[0][idx])
            })
        
        return gesture, float(confidence), top_predictions
    
    def smooth_prediction(self, current_gesture, confidence):
        self.prediction_history.append((current_gesture, confidence))
        
        if len(self.prediction_history) == self.prediction_history.maxlen:
            gestures = [gesture for gesture, conf in self.prediction_history 
                       if conf > self.confidence_threshold]
            if gestures:
                return max(set(gestures), key=gestures.count)
        
        return current_gesture if confidence > self.confidence_threshold else "Unknown"

gesture_tester = CustomGestureTester('custom_gesture_model.pkl')

frame_queue = queue.Queue(maxsize=2)
prediction_running = threading.Event()
prediction_running.set()

def update_latest_prediction(gesture, confidence, top_predictions):
    global latest_prediction
    latest_prediction = {
        "gesture": gesture,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "timestamp": datetime.now().isoformat()
    }

def prediction_processor():
    print("Prediction processor started")
    skip_counter = 0
    skip_frames = 2  # Process every 3rd frame to reduce load
    
    while prediction_running.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            skip_counter += 1
            
            if skip_counter < skip_frames:
                continue
            skip_counter = 0
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = gesture_tester.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks and predict gesture
                    landmarks = gesture_tester.extract_landmarks(hand_landmarks)
                    gesture, conf, top_preds = gesture_tester.predict_gesture(landmarks)
                    
                    # Apply smoothing
                    smoothed_gesture = gesture_tester.smooth_prediction(gesture, conf)
                    
                    # Update latest prediction
                    update_latest_prediction(smoothed_gesture, conf, top_preds)
                    break
            else:
                update_latest_prediction("No hand", 0.0, [])
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

def gen_frames():
    """Generate video frames only - no processing"""
    frame_count = 0
    while True:
        success, frame = video_camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        if frame_count % 3 == 0 and not frame_queue.full(): 
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
        
        frame_count += 1
        
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

prediction_thread = threading.Thread(target=prediction_processor, daemon=True)
prediction_thread.start()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gesture Recognition</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 30px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                color: white;
                font-size: 2.8rem;
                font-weight: 300;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }
            
            .header p {
                color: rgba(255, 255, 255, 0.85);
                font-size: 1.1rem;
                max-width: 600px;
                margin: 0 auto;
                line-height: 1.6;
            }
            
            .dashboard {
                display: grid;
                grid-template-columns: 1fr 400px;
                gap: 30px;
            }
            
            .video-container {
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                overflow: hidden;
            }
            
            .video-container h2 {
                color: #444;
                font-weight: 500;
                margin-bottom: 15px;
                padding-left: 5px;
            }
            
            .video-wrapper {
                border-radius: 12px;
                overflow: hidden;
                background: #f8f9fa;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 500px;
            }
            
            .video-wrapper img {
                max-width: 100%;
                height: auto;
                display: block;
            }
            
            .prediction-container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
            }
            
            .prediction-header {
                display: flex;
                align-items: center;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eaeaea;
            }
            
            .prediction-header i {
                font-size: 1.8rem;
                color: #667eea;
                margin-right: 12px;
            }
            
            .prediction-header h2 {
                color: #444;
                font-weight: 500;
            }
            
            .current-prediction {
                text-align: center;
                margin-bottom: 35px;
                padding: 25px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 15px;
            }
            
            .gesture-name {
                font-size: 2.2rem;
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
                min-height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .confidence-meter {
                margin: 20px 0;
            }
            
            .confidence-bar {
                height: 12px;
                background: #e0e0e0;
                border-radius: 6px;
                overflow: hidden;
                margin-bottom: 8px;
            }
            
            .confidence-fill {
                height: 100%;
                border-radius: 6px;
                background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
                width: 0%;
                transition: width 0.5s ease;
            }
            
            .confidence-text {
                font-size: 0.95rem;
                color: #666;
                display: flex;
                justify-content: space-between;
            }
            
            .top-predictions {
                margin-top: 30px;
            }
            
            .top-predictions h3 {
                color: #555;
                font-weight: 500;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }
            
            .top-predictions h3 i {
                margin-right: 10px;
                color: #764ba2;
            }
            
            .prediction-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                margin-bottom: 12px;
                background: #f8f9fa;
                border-radius: 10px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .prediction-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            }
            
            .prediction-name {
                font-weight: 500;
                color: #444;
            }
            
            .prediction-confidence {
                font-weight: 600;
                color: #667eea;
                font-size: 1.1rem;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                margin-top: 20px;
                padding: 12px;
                background: #f1f8ff;
                border-radius: 10px;
                font-size: 0.9rem;
                color: #0366d6;
            }
            
            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #6bcf7f;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .timestamp {
                color: #888;
                font-size: 0.85rem;
                text-align: center;
                margin-top: 15px;
            }
            
            .performance-info {
                margin-top: 20px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                font-size: 0.85rem;
                color: #666;
                text-align: center;
            }
            
            .footer {
                text-align: center;
                margin-top: 40px;
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.9rem;
            }
            
            @media (max-width: 1024px) {
                .dashboard {
                    grid-template-columns: 1fr;
                }
                
                .container {
                    padding: 20px;
                }
                
                .header h1 {
                    font-size: 2.2rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-hand-paper"></i> Gesture Recognition</h1>
                <p>Real-time hand gesture detection using MediaPipe and custom MLP model</p>
            </div>
            
            <div class="dashboard">
                <div class="video-container">
                    <h2><i class="fas fa-video"></i> Live Camera Feed</h2>
                    <div class="video-wrapper">
                        <img src="/video" alt="Live camera feed" id="video-feed">
                    </div>
                    <div class="performance-info">
                        <i class="fas fa-tachometer-alt"></i> 
                        High-speed video stream (no processing overlay)
                    </div>
                </div>
                
                <div class="prediction-container">
                    <div class="prediction-header">
                        <i class="fas fa-brain"></i>
                        <h2>Gesture Analysis</h2>
                    </div>
                    
                    <div class="current-prediction">
                        <div class="gesture-name" id="current-gesture">Waiting for hand...</div>
                        <div class="confidence-meter">
                            <div class="confidence-bar">
                                <div class="confidence-fill" id="confidence-fill"></div>
                            </div>
                            <div class="confidence-text">
                                <span>Low</span>
                                <span id="confidence-value">0.00</span>
                                <span>High</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="top-predictions">
                        <h3><i class="fas fa-chart-bar"></i> Top Predictions</h3>
                        <div id="top-predictions-list">
                            <div class="prediction-item">
                                <span class="prediction-name">Loading...</span>
                                <span class="prediction-confidence">0.00</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span id="status-text">System active. Show your hand to the camera.</span>
                    </div>
                    
                    <div class="timestamp" id="timestamp">
                        Last updated: --
                    </div>
                    <div class="performance-info">
                        <i class="fas fa-microchip"></i> 
                        Prediction running in separate thread (async)
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Model: Custom MLP | Gestures: <span id="model-gestures">""" + ', '.join(list(gesture_tester.label_encoder.classes_)) + """</span></p>
            </div>
        </div>
        
        <script>
            let updateInterval = 500; // Update prediction every 500ms
            let lastUpdateTime = 0;
            
            // Update prediction data
            async function updatePredictionData() {
                const now = Date.now();
                if (now - lastUpdateTime < updateInterval) {
                    return;
                }
                lastUpdateTime = now;
                
                try {
                    const response = await fetch('/prediction');
                    const data = await response.json();
                    
                    // Update current gesture
                    document.getElementById('current-gesture').textContent = data.gesture;
                    
                    // Update confidence meter
                    const confidence = data.confidence * 100;
                    document.getElementById('confidence-fill').style.width = confidence + '%';
                    document.getElementById('confidence-value').textContent = data.confidence.toFixed(2);
                    
                    // Update top predictions
                    const predictionsList = document.getElementById('top-predictions-list');
                    predictionsList.innerHTML = '';
                    
                    data.top_predictions.forEach(pred => {
                        const item = document.createElement('div');
                        item.className = 'prediction-item';
                        item.innerHTML = `
                            <span class="prediction-name">${pred.gesture}</span>
                            <span class="prediction-confidence">${pred.confidence.toFixed(2)}</span>
                        `;
                        predictionsList.appendChild(item);
                    });
                    
                    // Update timestamp
                    if (data.timestamp) {
                        const date = new Date(data.timestamp);
                        document.getElementById('timestamp').textContent = 
                            'Last updated: ' + date.toLocaleTimeString();
                    }
                    
                    // Update status
                    const statusText = document.getElementById('status-text');
                    if (data.gesture === "No hand") {
                        statusText.textContent = "Show your hand to the camera";
                    } else if (data.confidence > 0.8) {
                        statusText.textContent = "High confidence detection";
                    } else if (data.confidence > 0.6) {
                        statusText.textContent = "Medium confidence detection";
                    } else {
                        statusText.textContent = "Low confidence - try adjusting hand position";
                    }
                    
                } catch (error) {
                    console.error('Error fetching prediction data:', error);
                }
            }
            
            // Initial update
            updatePredictionData();
            
            // Update periodically
            setInterval(updatePredictionData, updateInterval);
            
            // Handle video feed errors
            document.getElementById('video-feed').onerror = function() {
                this.src = '/video?' + new Date().getTime();
            };
            
            // Preload prediction endpoint
            fetch('/prediction').catch(() => {});
        </script>
    </body>
    </html>
    """

@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/prediction")
def get_prediction():
    return latest_prediction

@app.on_event("shutdown")
def shutdown_event():
    global prediction_running
    prediction_running.clear()
    
    if video_camera.isOpened():
        video_camera.release()
    if prediction_camera.isOpened():
        prediction_camera.release()
    
    gesture_tester.hands.close()
    print("Cameras and gesture detector released")

if __name__ == "__main__":
    import uvicorn
    print("=== Gesture Recognition FastAPI Server ===")
    print("Server starting... Access the stream at http://localhost:8000")
    print(f"Available gestures: {list(gesture_tester.label_encoder.classes_)}")
    print("Video streaming and prediction processing are now separated for better performance")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")