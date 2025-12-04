import streamlit as st
import cv2
import numpy as np
from PIL import Image
import emotion_classifier as ec
import io
import wave
import base64
import time
import os
import sys

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
    # Note: many mobile browsers block autoplay. Prefer using `Enable Audio` first
    # which will preload an Audio element in the browser via an explicit user gesture.
    try:
        # If audio not enabled in session, render a play button to request gesture
        if not st.session_state.get("audio_enabled", False):
            if st.button("▶️ Enable audio to allow sleep sound (tap once)"):
                st.session_state["audio_enabled"] = True
                # After enabling, we inject a small JS helper to create a global audio element
                _inject_preloaded_audio()
            return

        # If enabled, try to play preloaded audio element in browser via JS
        # Prefer the repo MP3 if available, otherwise fallback to generated WAV.
        audio_bytes = None
        audio_fmt = "audio/mp3"
        if os.path.exists("sleep.mp3"):
            try:
                with open("sleep.mp3", "rb") as f:
                    audio_bytes = f.read()
                    audio_fmt = "audio/mp3"
            except Exception:
                audio_bytes = None

        if not audio_bytes:
            audio_bytes = generate_tone_wav_bytes(duration_s=3.0, freq=330.0)
            audio_fmt = "audio/wav"

        _play_bytes_via_js(audio_bytes, audio_fmt)
    except Exception:
        st.warning(f"⚠️ Audio playback unavailable")


def speak_joke_html(joke_text: str):
    """Use browser Web Speech API to speak a joke via injected HTML/JS."""
    try:
        if not st.session_state.get("audio_enabled", False):
            # Show a button so user can explicitly trigger the joke TTS
            if st.button("🔊 Play joke aloud (requires tapping to enable audio)"):
                st.session_state["audio_enabled"] = True
                _inject_preloaded_audio()
                # proceed to speak after enabling
            else:
                return

        # Escape special chars safely for JS string
        escaped = joke_text.replace("\n", " ").replace("\\", " ").replace('"', "'").strip()
        if not escaped or len(escaped) < 3:
            return  # Skip if too short

        html = f"""
        <script>
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {{
            try {{
                const msg = new SpeechSynthesisUtterance("{escaped}");
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.volume = 0.9;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(msg);
            }} catch(err) {{
                console.warn("TTS failed: " + err.message);
            }}
        }}
        </script>
        """
        components.html(html, height=0)
    except Exception:
        pass  # Silent fail for TTS — not critical


def _play_bytes_via_js(audio_bytes: bytes, audio_fmt: str = "audio/mp3"):
    """Inject JS that plays provided audio bytes (base64) using a preloaded Audio element.
    This relies on the user having already pressed an explicit 'Enable audio' button in the UI.
    """
    try:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        # Create a small JS snippet that looks for `window._sleepAudio` and uses it,
        # otherwise creates a new audio element and plays it.
        html = f"""
        <script>
        (function() {{
            try {{
                const src = 'data:{audio_fmt};base64,{b64}';
                let a = window._sleepAudio;
                if (!a) {{
                    a = new Audio();
                    a.src = src;
                    a.preload = 'auto';
                    window._sleepAudio = a;
                }} else {{
                    // Replace src to ensure latest data
                    a.src = src;
                }}
                a.play().catch(err => console.warn('play failed', err));
            }} catch(err) {{ console.warn('audio play error', err); }}
        }})();
        </script>
        """
        components.html(html, height=0)
    except Exception:
        # Fallback to server-side st.audio (may be blocked on mobile)
        try:
            st.audio(audio_bytes, format=audio_fmt)
        except Exception:
            pass


def _inject_preloaded_audio():
    """Create a small global audio element in the browser (requires user gesture).
    We try to preload `sleep.mp3` if present; otherwise preload a short tone.
    """
    try:
        audio_bytes = None
        audio_fmt = "audio/mp3"
        if os.path.exists("sleep.mp3"):
            try:
                with open("sleep.mp3", "rb") as f:
                    audio_bytes = f.read()
                    audio_fmt = "audio/mp3"
            except Exception:
                audio_bytes = None

        if not audio_bytes:
            audio_bytes = generate_tone_wav_bytes(duration_s=1.5, freq=440.0)
            audio_fmt = "audio/wav"

        b64 = base64.b64encode(audio_bytes).decode("ascii")
        html = f"""
        <script>
        (function(){{
            try {{
                const src = 'data:{audio_fmt};base64,{b64}';
                let a = new Audio();
                a.src = src;
                a.preload = 'auto';
                window._sleepAudio = a;
                window._audioEnabled = true;
            }} catch(err) {{ console.warn('preload audio failed', err); }}
        }})();
        </script>
        """
        components.html(html, height=0)
    except Exception:
        pass


def process_np_image(np_img, image_size):
    """Run mediapipe on an RGB numpy array and return (emotion_label, display_img, debug_info)."""
    try:
        # Validate input
        if np_img is None or np_img.size == 0:
            return "No face", np_img, {}
        
        if len(np_img.shape) < 2:
            return "No face", np_img, {}
        
        # Resize for performance
        h, w = np_img.shape[:2]
        if w <= 0 or h <= 0:
            return "No face", np_img, {}
        
        aspect = h / max(w, 1)
        new_h = max(1, int(image_size * aspect))
        display_img = cv2.resize(np_img, (image_size, new_h))
        
        # Run detection
        res = ec.facemesh.process(display_img)
        debug_info = {}
        emotion_label = "No face"
        
        if res.multi_face_landmarks and len(res.multi_face_landmarks) > 0:
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
    except Exception as e:
        st.error(f"Image processing error: {str(e)[:80]}")
        return "No face", np_img, {}


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

        # Global audio enable button for mobile: user should tap this once to allow autoplay/TTS
        audio_col1, audio_col2 = st.columns([1, 3])
        with audio_col1:
            if not st.session_state.get("audio_enabled", False):
                if st.button("▶️ Enable audio for sleep sound & jokes"):
                    st.session_state["audio_enabled"] = True
                    _inject_preloaded_audio()
            else:
                st.write("Audio: Enabled ✅")
        with audio_col2:
            st.markdown("Small note: On mobile, please tap 'Enable audio' once so the app can play sounds and speak jokes.")

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
                from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
                import av
            except ImportError as e:
                st.error("❌ streamlit-webrtc or av not installed. Please redeploy or run: `pip install streamlit-webrtc av`")
                st.stop()

            class EmotionProcessor(VideoProcessorBase):
                def __init__(self):
                    self.frame = None
                    self.latest_emotion = "No face"

                def recv(self, frame):
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        if img is None or img.size == 0:
                            return frame
                        
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
                    except Exception as e:
                        # Silently return original frame on error
                        return frame

            ctx = webrtc_streamer(key="live", video_processor_factory=EmotionProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

            # Auto-capture controls
            auto_col1, auto_col2 = st.columns([1, 1])
            with auto_col1:
                auto_capture = st.checkbox("🔁 Auto-capture from live (periodic)", value=False)
            with auto_col2:
                auto_interval = st.slider("Interval (s)", 1, 10, 3)

            # If auto-capture is enabled, inject JS that clicks the capture button periodically
            if auto_capture:
                js = f"""
                <script>
                (function(){{
                    try {{
                        // Periodically look for the Streamlit button with the capture label and click it
                        const label = '📸 Capture snapshot from live';
                        function clickIfFound() {{
                            const buttons = Array.from(document.querySelectorAll('button'));
                            for (const b of buttons) {{
                                if ((b.innerText || b.textContent || '').trim().includes(label)) {{
                                    try {{ b.click(); }} catch(e) {{ console.warn('click failed', e); }}
                                    return;
                                }}
                            }}
                        }}
                        // Initial try and then interval
                        clickIfFound();
                        window._autoCaptureInterval = window._autoCaptureInterval || setInterval(clickIfFound, {auto_interval} * 1000);
                    }} catch(err) {{ console.warn('auto-capture setup failed', err); }}
                }})();
                </script>
                """
                components.html(js, height=0)

            # Controls to capture current live frame and trigger actions
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("📸 Capture snapshot from live"):
                    if ctx and ctx.video_processor and hasattr(ctx.video_processor, 'frame') and ctx.video_processor.frame is not None:
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
            try:
                # Read image
                image = Image.open(uploaded_file)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                img_array = np.array(image)
                
                # Validate image
                if img_array is None or img_array.size == 0:
                    st.error("❌ Invalid image file")
                    return
                
                # Convert PIL RGB to OpenCV BGR if needed
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    rgb_img = img_array
                    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif len(img_array.shape) == 2:
                    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    bgr_img = img_array
                else:
                    st.error(f"❌ Unsupported image shape: {img_array.shape}")
                    return
                
                # Resize for display
                display_img = cv2.resize(rgb_img, (320, int(rgb_img.shape[0] * 320 / max(rgb_img.shape[1], 1))))
                
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
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)[:100]}")


if __name__ == "__main__":
    main()
