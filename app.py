import streamlit as st
import cv2
import numpy as np
from PIL import Image
import emotion_classifier as ec
import io
import wave

# --- Small utilities: tone generator and jokes list ---
def generate_tone_wav_bytes(duration_s=2.0, freq=440.0, sr=22050):
    """Generate a WAV (PCM16) bytes for a sine tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    # convert to 16-bit PCM
    audio = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()

# A short list of jokes to display when person looks sad
JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "I used to play piano by ear, but now I use my hands.",
    "Why don't eggs tell jokes? They'd crack each other up.",
]

import os
import streamlit.components.v1 as components


def play_sleep_audio():
    """Play sleep audio from repo `sleep.mp3` if present, otherwise play generated tone."""
    if os.path.exists("sleep.mp3"):
        with open("sleep.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
    else:
        tone = generate_tone_wav_bytes(duration_s=3.0, freq=330.0)
        st.audio(tone, format="audio/wav")


def speak_joke_html(joke_text: str):
    """Use browser Web Speech API to speak a joke via injected HTML/JS."""
    escaped = joke_text.replace("\n", "\\n").replace("\"", "\\\"")
    html = f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{escaped}");
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(html, height=0)


def process_np_image(np_img, image_size):
    """Run mediapipe on an RGB numpy array and return (emotion_label, display_img, debug_info)."""
    # Resize for performance
    display_img = cv2.resize(np_img, (image_size, int(np_img.shape[0] * image_size / np_img.shape[1])))
    # Run detection
    res = ec.facemesh.process(display_img)
    debug_info = {}
    emotion_label = "No face"
    if res.multi_face_landmarks:
        face_lms = res.multi_face_landmarks[0]
        landmarks = face_lms.landmark
        # Draw face mesh
        ec.mpdrawing.draw_landmarks(
            image=display_img,
            landmark_list=face_lms,
            connections=ec.mpface.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=ec.mpstyles.get_default_face_mesh_tesselation_style()
        )
        mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt = ec.get_features(landmarks)
        emotion = ec.classify_emotion(mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt)
        cry_state = ec.classify_crying(eye_open, inner_eye_dist, 0)
        emotion_label = cry_state if cry_state else emotion
        debug_info = {
            "mouth_open": f"{mouth_open:.3f}",
            "mouth_width": f"{mouth_width:.3f}",
            "eye_open": f"{eye_open:.3f}",
            "nose_cod": f"{nose_cod:.3f}",
            "inner_eye_dist": f"{inner_eye_dist:.3f}",
            "lip_asym": f"{lip_asym:.3f}",
            "head_tilt": f"{head_tilt:.3f}",
        }
    return emotion_label, display_img, debug_info


def main():
    st.set_page_config(page_title="Emotion Classifier", layout="wide")
    
    st.title("🎭 Real-Time Emotion Classifier")
    st.markdown("Detect emotions using facial landmarks and MediaPipe")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    frame_interval = st.sidebar.slider("Frame interval (ms)", 100, 2000, 500)
    show_debug = st.sidebar.checkbox("Show debug info", value=True)
    image_size = st.sidebar.slider("Resize width (px)", 160, 640, 320)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📷 Live Detection", "📤 Upload Image"])
    
    with tab1:
        st.subheader("Live Webcam Detection")
        st.write("Choose mode: Snapshot (quick) or Live (real-time via WebRTC).")

        mode = st.radio("Mode", ["Snapshot", "Live (webrtc)"], index=0, horizontal=True)

        # Session state defaults
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False
        if "joke_idx" not in st.session_state:
            st.session_state.joke_idx = 0
        if "last_emotion" not in st.session_state:
            st.session_state.last_emotion = None
        if "last_sleep_play" not in st.session_state:
            st.session_state.last_sleep_play = 0.0

        # Start / Stop buttons
        col_start, col_stop = st.columns([1, 1])
        with col_start:
            if st.button("📹 Start Camera", key="start_cam"):
                st.session_state.camera_running = True
        with col_stop:
            if st.button("⏹️ Stop Camera", key="stop_cam"):
                st.session_state.camera_running = False

        if mode == "Snapshot":
            # Show camera input only when running
            if st.session_state.camera_running:
                img_file = st.camera_input("Take a picture")

                if img_file is not None:
                    # Read image from camera input
                    image = Image.open(img_file).convert("RGB")
                    img_array = np.array(image)

                    # Process the RGB numpy image
                    emotion_label, display_img, debug_info = process_np_image(img_array, image_size)

                    if emotion_label != "No face":
                        # Show results with improved UI
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.image(display_img, caption="Detected Emotion", channels="RGB")

                        with col2:
                            st.markdown(f"### 🎯 Emotion: **{emotion_label}**")
                            if show_debug and debug_info:
                                for k, v in debug_info.items():
                                    st.write(f"- {k}: {v}")

                        # Actions based on emotion
                        import time
                        if emotion_label.lower() in ["sleepy", "drowsy", "yawning"]:
                            now = time.time()
                            if now - st.session_state.last_sleep_play > 5.0:
                                play_sleep_audio()
                                st.session_state.last_sleep_play = now

                        if emotion_label.lower() == "sad":
                            idx = st.session_state.joke_idx % len(JOKES)
                            joke = JOKES[idx]
                            st.info(f"Here's a joke to cheer you up: \n\n{joke}")
                            speak_joke_html(joke)
                            if st.session_state.last_emotion != "sad":
                                st.session_state.joke_idx += 1

                        st.session_state.last_emotion = emotion_label.lower()
                    else:
                        st.warning("⚠️ No face detected in the image. Please try again.")
                else:
                    st.info("Click 'Take a picture' to capture a frame.")

        else:
            # Live mode via streamlit-webrtc
            try:
                from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
                import av
            except Exception as e:
                st.warning("streamlit-webrtc is not installed in the environment. Install with `pip install streamlit-webrtc av` and redeploy.")
                st.stop()

            class EmotionProcessor(VideoProcessorBase):
                def __init__(self):
                    self.frame = None
                    self.latest_emotion = "No face"

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    emotion_label, display_img, _ = process_np_image(rgb, image_size)
                    # process_np_image returns RGB display_img; convert to BGR for output
                    out = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                    # overlay label
                    color = ec.COLOR_MAP.get(emotion_label, (0, 255, 0))
                    cv2.rectangle(out, (0, 0), (out.shape[1], 55), (0, 0, 0), -1)
                    cv2.putText(out, emotion_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    self.latest_emotion = emotion_label
                    self.frame = out
                    return av.VideoFrame.from_ndarray(out, format="bgr24")

            ctx = webrtc_streamer(key="live", video_processor_factory=EmotionProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

            # Controls to capture current live frame and trigger actions
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("📸 Capture snapshot from live"):
                    if ctx and ctx.video_processor and ctx.video_processor.frame is not None:
                        frame_bgr = ctx.video_processor.frame
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        emotion_label, display_img, debug_info = process_np_image(rgb, image_size)

                        if emotion_label != "No face":
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(display_img, caption="Detected Emotion (live snapshot)", channels="RGB")
                            with col2:
                                st.markdown(f"### 🎯 Emotion: **{emotion_label}**")
                                if show_debug and debug_info:
                                    for k, v in debug_info.items():
                                        st.write(f"- {k}: {v}")

                            # Actions
                            import time
                            if emotion_label.lower() in ["sleepy", "drowsy", "yawning"]:
                                now = time.time()
                                if now - st.session_state.last_sleep_play > 5.0:
                                    play_sleep_audio()
                                    st.session_state.last_sleep_play = now

                            if emotion_label.lower() == "sad":
                                idx = st.session_state.joke_idx % len(JOKES)
                                joke = JOKES[idx]
                                st.info(f"Here's a joke to cheer you up: \n\n{joke}")
                                speak_joke_html(joke)
                                if st.session_state.last_emotion != "sad":
                                    st.session_state.joke_idx += 1

                            st.session_state.last_emotion = emotion_label.lower()
                        else:
                            st.warning("⚠️ No face detected in the live snapshot. Try again.")
                    else:
                        st.warning("No live frame available yet. Allow camera and wait a moment.")

            with col_b:
                if ctx and ctx.video_processor:
                    st.metric("Live emotion", ctx.video_processor.latest_emotion)
                else:
                    st.info("Live stream not started yet.")

    with tab2:
        st.subheader("Upload an Image")
        st.write("Upload a photo to detect emotions in a single frame.")
        
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert PIL RGB to OpenCV BGR if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                rgb_img = img_array
                bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                bgr_img = img_array
            
            # Resize for display
            display_img = cv2.resize(rgb_img, (320, int(rgb_img.shape[0] * 320 / rgb_img.shape[1])))
            
            # Run detection
            res = ec.facemesh.process(display_img)
            
            if res.multi_face_landmarks:
                face_lms = res.multi_face_landmarks[0]
                landmarks = face_lms.landmark
                
                # Draw face mesh
                ec.mpdrawing.draw_landmarks(
                    image=display_img,
                    landmark_list=face_lms,
                    connections=ec.mpface.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=ec.mpstyles.get_default_face_mesh_tesselation_style()
                )
                
                # Get features and classify
                mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt = ec.get_features(landmarks)
                emotion = ec.classify_emotion(mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt)
                cry_state = ec.classify_crying(eye_open, inner_eye_dist, 0)
                emotion_label = cry_state if cry_state else emotion
                
                # Draw label
                color = ec.COLOR_MAP.get(emotion_label, (0, 255, 0))
                cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 55), (0, 0, 0), -1)
                cv2.putText(
                    display_img,
                    emotion_label,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )
                
                # Display result
                col1, col2 = st.columns(2)
                with col1:
                    st.image(display_img, caption="Detected Emotion", channels="BGR")
                
                with col2:
                    st.subheader(f"Emotion: {emotion_label}")
                    st.write("**Debug Info:**")
                    st.write(f"- Eye Open: {eye_open:.3f}")
                    st.write(f"- Mouth Open: {mouth_open:.3f}")
                    st.write(f"- Mouth Width: {mouth_width:.3f}")
                    st.write(f"- Nose Cod: {nose_cod:.3f}")
                    st.write(f"- Inner Eye Dist: {inner_eye_dist:.3f}")
                    st.write(f"- Lip Asymmetry: {lip_asym:.3f}")
                    st.write(f"- Head Tilt: {head_tilt:.3f}")
            else:
                st.warning("⚠️ No face detected in the image. Please try another image.")
                st.image(rgb_img, caption="Uploaded Image", use_container_width=True)


if __name__ == "__main__":
    main()
