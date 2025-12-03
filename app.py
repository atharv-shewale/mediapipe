import streamlit as st
import cv2
import numpy as np
from PIL import Image
import emotion_classifier as ec


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
        st.write("Uses your device's camera to detect emotions in real-time.")
        
        # Initialize session state for webcam
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False
        
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("📹 Start Camera", key="start_cam")
        with col2:
            stop_btn = st.button("⏹️ Stop Camera", key="stop_cam")
        
        if start_btn:
            st.session_state.camera_running = True
        if stop_btn:
            st.session_state.camera_running = False
        
        # Placeholder for video display
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Unable to access webcam. Please check camera permissions.")
            else:
                st.success("✅ Camera connected. Detecting emotions...")
                
                # Control frame interval
                frame_count = 0
                
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame.")
                        break
                    
                    # Resize for performance
                    h, w = frame.shape[:2]
                    scale = image_size / w
                    new_h = int(h * scale)
                    frame_resized = cv2.resize(frame, (image_size, new_h))
                    
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # Run face detection
                    res = ec.facemesh.process(rgb_frame)
                    
                    # Draw landmarks and get emotion
                    frame_display = frame_resized.copy()
                    emotion_label = "No face"
                    debug_info = {}
                    
                    if res.multi_face_landmarks:
                        face_lms = res.multi_face_landmarks[0]
                        landmarks = face_lms.landmark
                        
                        # Draw face mesh
                        ec.mpdrawing.draw_landmarks(
                            image=frame_display,
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
                        
                        debug_info = {
                            "mouth_open": f"{mouth_open:.3f}",
                            "mouth_width": f"{mouth_width:.3f}",
                            "eye_open": f"{eye_open:.3f}",
                            "nose_cod": f"{nose_cod:.3f}",
                            "inner_eye_dist": f"{inner_eye_dist:.3f}",
                            "lip_asym": f"{lip_asym:.3f}",
                            "head_tilt": f"{head_tilt:.3f}",
                        }
                    
                    # Draw label with background
                    color = ec.COLOR_MAP.get(emotion_label, (0, 255, 0))
                    cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], 55), (0, 0, 0), -1)
                    cv2.putText(
                        frame_display,
                        emotion_label,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        2,
                    )
                    
                    # Display in Streamlit
                    video_placeholder.image(frame_display, channels="BGR", use_container_width=True)
                    
                    # Show debug info
                    if show_debug and debug_info:
                        with stats_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Emotion", emotion_label)
                            with col2:
                                st.metric("Eye Open", debug_info.get("eye_open", "N/A"))
                            with col3:
                                st.metric("Mouth Open", debug_info.get("mouth_open", "N/A"))
                    
                    # Frame interval control
                    frame_count += 1
                    if frame_count * 33 > frame_interval:  # ~30 FPS base
                        frame_count = 0
                        # Allow Streamlit to process reruns
                
                cap.release()
        else:
            st.info("Click 'Start Camera' to begin emotion detection.")
    
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
