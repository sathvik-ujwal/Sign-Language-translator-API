import cv2
import mediapipe as mp

def process_video(video_capture):
    """
    Process a video file to extract keypoints using MediaPipe.

    Args:
        video_capture (cv2.VideoCapture): OpenCV VideoCapture object.

    Returns:
        List[List[List[float]]]: Keypoints for each frame.
    """
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    keypoints_sequence = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 keypoints as (x, y, z)
                hand_keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                frame_keypoints.append(hand_keypoints)

        keypoints_sequence.append(frame_keypoints)

    video_capture.release()
    return keypoints_sequence
