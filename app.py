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
        st.subheader("Live Webcam Detection (Snapshot mode)")
        st.write("Uses your device's camera to take snapshots and detect emotions. For continuous live video, consider using the streamlit-webrtc option.")

        # Use client-side camera input (works on deployed HTTPS sites)
        img_file = st.camera_input("Take a picture")

        if img_file is not None:
            # Read image from camera input
            image = Image.open(img_file).convert("RGB")
            img_array = np.array(image)

            # Convert PIL RGB to OpenCV BGR if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                rgb_img = img_array
                bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                bgr_img = img_array

            # Resize for performance
            display_img = cv2.resize(rgb_img, (image_size, int(rgb_img.shape[0] * image_size / rgb_img.shape[1])))

            # Run detection using MediaPipe (expects RGB)
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

                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(display_img, caption="Detected Emotion", channels="RGB")

                with col2:
                    st.subheader(f"Emotion: {emotion_label}")
                    if show_debug:
                        st.write("**Debug Info:**")
                        st.write(f"- Eye Open: {eye_open:.3f}")
                        st.write(f"- Mouth Open: {mouth_open:.3f}")
                        st.write(f"- Mouth Width: {mouth_width:.3f}")
                        st.write(f"- Nose Cod: {nose_cod:.3f}")
                        st.write(f"- Inner Eye Dist: {inner_eye_dist:.3f}")
                        st.write(f"- Lip Asymmetry: {lip_asym:.3f}")
                        st.write(f"- Head Tilt: {head_tilt:.3f}")
            else:
                st.warning("⚠️ No face detected in the image. Please try again.")
    
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
