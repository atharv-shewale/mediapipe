import cv2
import mediapipe as mp
import numpy as np

#initiliaze mp
mpface = mp.solutions.face_mesh
mpdrawing = mp.solutions.drawing_utils
mpstyles = mp.solutions.drawing_styles

facemesh = mpface.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_features(landmarks):
    coords = np.array([[p.x, p.y] for p in landmarks])
    # Normalize
    xs = coords[:, 0]
    ys = coords[:, 1]
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)

    def dist(i, j):
        return np.linalg.norm((coords[i] - coords[j]) / np.array([width, height]))

    # Mouth
    upper_lip = 13
    lower_lip = 14
    left_mouth_corner = 61
    right_mouth_corner = 291

    # Eye
    eye_outer = 33
    eye_inner = 133
    eye_top = 159
    eye_bottom = 145

    mouth_open = dist(upper_lip, lower_lip)
    mouth_width = dist(left_mouth_corner, right_mouth_corner)

    eye_vertical = dist(eye_top, eye_bottom)
    eye_horizontal = dist(eye_outer, eye_inner)
    eye_open = eye_vertical / (eye_horizontal + 1e-6)

    # Nose (for head up/down)
    nose_idx = 1
    nose = coords[nose_idx]
    center = coords.mean(axis=0)
    nose_cod = (nose[1] - center[1]) / height

    # Crying feature (squeezing between inner corners)
    inner_eye_dist = dist(133, 362)

    # NEW: lip asymmetry (for disgust) – vertical difference between mouth corners
    left_lip_y = coords[left_mouth_corner, 1]
    right_lip_y = coords[right_mouth_corner, 1]
    lip_asym = (left_lip_y - right_lip_y) / height  # positive/negative = tilt

    
    left_side = coords[234, 1]   # approx left cheek/temple
    right_side = coords[454, 1]  # approx right cheek/temple
    head_tilt = (left_side - right_side) / height  # +ve / -ve tilt

    return (
        mouth_open,
        mouth_width,
        eye_open,
        nose_cod,
        inner_eye_dist,
        lip_asym,
        head_tilt,
    )


def classify_crying(eye_open, inner_eye_dist, blink_counter):
    
    if eye_open < 0.16 and inner_eye_dist < 0.28 and blink_counter >= 4:
        return "crying / watery eyes"
    else:
        return None


def classify_emotion(
    mouth_open,
    mouth_width,
    eye_open,
    nose_cod,
    inner_eye_dist,
    lip_asym,
    head_tilt,
):
    """
    Rule-based emotion classification using geometric facial features.
    We keep your original ideas and extend them with more nuanced rules.
    """

    
    big_mouth_open = mouth_open > 0.18        # strong open
    strong_smile = mouth_width > 0.45 and eye_open > 0.18
    relaxed_smile = mouth_width > 0.045 and eye_open > 0.17
    eyes_very_open = eye_open > 0.30
    eyes_narrow = eye_open < 0.18
    eyes_very_narrow = eye_open < 0.14
    head_down = nose_cod > 0.03
    head_up = nose_cod < -0.03
    strong_head_tilt = abs(head_tilt) > 0.05
    lip_asym_strong = abs(lip_asym) > 0.03

   
    if big_mouth_open and eyes_narrow:
        return "yawning / drowsy"

    
    if eyes_very_open and mouth_open > 0.08:
        return "shocked"

    
    if strong_smile:
        return "smile"

    
    if mouth_open > 0.10 and eye_open > 0.25:
        return "surprised"

    
    if mouth_width > 0.39 and eye_open >= 0.17:
        return "happy"

 
    if 0.15 <= eye_open < 0.20 and mouth_width < 0.04 and not big_mouth_open:
        return "sad"

    
    if eyes_narrow and inner_eye_dist < 0.27 and mouth_width < 0.04 and not big_mouth_open:
        return "angry"

    
    if lip_asym_strong and mouth_open < 0.06:
        return "disgust"

  
    if eyes_narrow:
        # keep your specific combined case
        if head_down and eyes_narrow:
            return "sleepy and looking down"
        if head_up and eyes_narrow:
            return "sleepy "
        return "sleepy"

    
    if head_down and 0.18 <= eye_open <= 0.26 and mouth_open < 0.06:
        return "thinking"

   
    if strong_head_tilt and eye_open > 0.18:
        return "confused"

    # Fallback
    return "neutral"


# -------- Dynamic label colors (UI-only, no logic change) --------
COLOR_MAP = {
    "happy": (0, 255, 255),                     # yellow
    "smile": (0, 200, 255),                     # softer yellow
    "surprised": (255, 0, 255),                 # pink
    "shocked": (0, 0, 255),                     # red
    "sleepy": (0, 128, 255),                    # orange
    "sleepy and looking down": (0, 140, 255),
    "sleepy ": (0, 128, 255),                   # your "sleepy " label
    "yawning / drowsy": (0, 165, 255),          # dark orange
    "crying / watery eyes": (255, 0, 0),        # blue-ish
    "sad": (255, 255, 255),                     # white
    "angry": (0, 0, 255),                       # red
    "disgust": (128, 0, 128),                   # purple
    "thinking": (255, 255, 0),                  # cyan
    "confused": (255, 255, 0),                  # cyan
    "neutral": (200, 200, 200),                 # light gray
    "No face": (100, 100, 100),                 # darker gray
}
# --------------------------------------------------------


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error in opening camers")
        return

    print("Cmaera is opend \n to exit press Ctrl+C")

    blink_counter = 0
    eye_closed_frames = 0
    EYE_CLOSE_THRESH = 0.14
    EYE_CLOSE_CONSEC_FRAMES = 2

    # ensure facemesh is initialized for the interactive main loop
    facemesh = get_facemesh()

    print("Facemesh initialized.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = facemesh.process(rgb_frame)

        if res.multi_face_landmarks:
            face_lms = res.multi_face_landmarks[0]
            landmarks = face_lms.landmark

            mpdrawing.draw_landmarks(
                image=frame,
                landmark_list=face_lms,
                connections=mpface.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpstyles.get_default_face_mesh_tesselation_style()
            )

            # unpack features (now 7 values)
            mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt = get_features(landmarks)

            emotion = classify_emotion(
                mouth_open,
                mouth_width,
                eye_open,
                nose_cod,
                inner_eye_dist,
                lip_asym,
                head_tilt,
            )
            cry_state = classify_crying(eye_open, inner_eye_dist, blink_counter)

            if cry_state:
                label = cry_state
            else:
                label = emotion

            # Debug info
            debug2 = (
                f"eye_open={eye_open:.3f} | inner_eye={inner_eye_dist:.3f} | "
                f"mouth_open={mouth_open:.3f} | mouth_width={mouth_width:.3f}"
            )
            cv2.putText(
                frame,
                debug2,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            debug_text = f"nose={nose_cod:.3f} | lip_asym={lip_asym:.3f} | tilt={head_tilt:.3f}"
            cv2.putText(
                frame,
                debug_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        else:
            label = "No face"

        # ------- Attractive overlay for main label -------
        color = COLOR_MAP.get(label, (0, 255, 0))

        # top bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)

        # shadow
        shadow_offset = 2
        cv2.putText(
            frame,
            label,
            (10 + shadow_offset, 40 + shadow_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
        )

        # main colored label
        cv2.putText(
            frame,
            label,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )
        # -------------------------------------------------

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
