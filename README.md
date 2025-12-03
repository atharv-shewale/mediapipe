# Emotion Classifier — Streamlit Web App

A real-time emotion detection webapp built with Streamlit and MediaPipe. Detects emotions from facial landmarks.

## Features
- 📷 **Live webcam detection** — real-time emotion recognition.
- 📤 **Image upload** — analyze still photos.
- 🎨 **Beautiful UI** — Streamlit-based interface.
- 📊 **Debug info** — view facial feature metrics.

## Quick Start (Local)

### 1. Install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the app
```powershell
streamlit run app.py
```
This opens the app at `http://localhost:8501` in your browser.

### 3. Use the app
- **Live Detection tab**: Click "Start Camera" to capture webcam frames and see real-time emotion labels.
- **Upload Image tab**: Upload a photo to detect emotion in a single frame.

## Emotions Detected
- happy, smile, surprised, shocked
- sleepy, sleepy and looking down, yawning / drowsy
- crying / watery eyes, sad, angry, disgust
- thinking, confused, neutral
- No face (if no face detected)

## Deploy

### Docker (Local Test)
```powershell
docker build -t emotion-classifier:local .
docker run --rm -p 8501:8501 emotion-classifier:local
```
Then visit `http://127.0.0.1:8501`.

### Streamlit Cloud (Free, Easiest)
1. Push repo to GitHub.
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud) → New app.
3. Connect your GitHub repo → select `app.py` → deploy.

### Render (Docker)
1. Push repo to GitHub.
2. Create Render Web Service from repo.
3. Set Environment = Docker.
4. Render builds and deploys; visit the auto-generated URL.

### Heroku (Container)
```bash
heroku container:login
heroku create my-emotion-app
docker build -t registry.heroku.com/my-emotion-app/web .
docker push registry.heroku.com/my-emotion-app/web
heroku container:release web -a my-emotion-app
```

### Local VPS / Ubuntu
```bash
sudo apt update && sudo apt install -y python3-venv ffmpeg libglib2.0-0 libgl1 libgomp1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Notes
- Webcam capture requires browser permission (handled automatically on first use).
- For best performance, use a modern browser (Chrome, Firefox, Edge).
- Frame interval can be adjusted in the sidebar.
- Requires a GPU or good CPU for smooth real-time detection (2GB+ RAM recommended).

## Files
- `app.py` — Streamlit app (main entry point).
- `emotion_classifier.py` — emotion detection logic (facial landmark extraction + classification).
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container image definition.
- `.dockerignore` — files to exclude from Docker image.
