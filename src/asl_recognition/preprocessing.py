import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

keypoint_map = {
    0: 0,  # Nose
    # 1, Neck
    12: 2,  # Right shoulder
    14: 3,  # Right elbow
    16: 4,   # Right wrist
    11: 5,  # Left shoulder
    13: 6,  # Left elbow
    15: 7,  # Left wrist
    # 8, Mid Hip
    5: 15,  # Right eye
    2: 16,  # Left eye
    8: 17,  # Right ear 
    7: 18  # Left ear
}

skeleton = [
    (0, 1),  # Nose to Neck
    (1, 2),  # Neck to Right Shoulder
    (2, 3),  # Right Shoulder to Right Elbow
    (3, 4),  # Right Elbow to Right Wrist
    (1, 5),  # Neck to Left Shoulder
    (5, 6),  # Left Shoulder to Left Elbow
    (6, 7),  # Left Elbow to Left Wrist
    (1, 8),  # Neck to Mid Hip
    (0, 16), # Nose to Left eye
    (0, 15), # Nose to Right eye
    (16, 18), # Left eye to left ear
    (15, 17)  # Right eye to right ear
]

def get_keypoints(landmarks, keypoint_map):
    keypoints = []

    for index, model_index in keypoint_map.items():
        if index < len(landmarks):
            lm = landmarks[index]
            keypoints.extend([lm.x, lm.y])
        else:
            keypoints.extend([0.0, 0.0])  # Pad with zeros
    return np.array(keypoints, dtype=np.float32).reshape(-1, 2)

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints[:, 0] = 2 * ((keypoints[:, 0] * image_width) / image_width - 0.5)  # Normalize x
    keypoints[:, 1] = 2 * ((keypoints[:, 1] * image_height) / image_height - 0.5)  # Normalize y
    return keypoints


def preprocess_frames(results):
    """
    Extracts keypoints and calculates specific points from the results of the MediaPipe Holistic model.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    try:
        points = {}
        keypoints = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate Neck and Mid Hip
            neck_x = (landmarks[11].x + landmarks[12].x) / 2
            neck_y = (landmarks[11].y + landmarks[12].y) / 2
            mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

            pose_keypoints = get_keypoints(landmarks, keypoint_map)

            # Add neck and mid-hip to the pose keypoints
            neck_point = [neck_x, neck_y ]
            mid_hip_point = [mid_hip_x , mid_hip_y]
            pose_keypoints = np.insert(pose_keypoints, [1, 8], [neck_point, mid_hip_point], axis=0)

            # Normalize keypoints to [-1, 1]
            # pose_keypoints = 2 * ((torch.FloatTensor(pose_keypoints) / torch.FloatTensor([256, 256])) - 0.5)
            pose_keypoints = normalize_keypoints(pose_keypoints, image_width=256, image_height=256)


            # Save points for visualization
            for idx, (x, y) in enumerate(pose_keypoints):
                points[idx] = (int(x), int(y))

            keypoints.append(pose_keypoints)
            #print("pose landmarks")
        else:
            # Pad with zeros if pose landmarks are missing
            keypoints.append(np.zeros((13, 2), dtype=np.float32))

        # Process left hand landmarks
        if results.left_hand_landmarks:
            left_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            left_hand_keypoints = normalize_keypoints(left_hand_keypoints, image_width=256, image_height=256)
            keypoints.append(left_hand_keypoints)
            #print("left hand landmarks")

        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))        
    
        # Process right hand landmarks
        if results.right_hand_landmarks:
            right_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            right_hand_keypoints = normalize_keypoints(right_hand_keypoints, image_width=256, image_height=256)  # Apply normalization
            keypoints.append(right_hand_keypoints)
            #print("right hand landmarks")
        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))
            
        combined_keypoints = np.concatenate(keypoints).flatten() if keypoints else None
        
        return points, combined_keypoints
    except Exception as e:
        print(f"Failed to preprocess frame: {str(e)}")
        return {}, None


def capture_frames(frame_buffer, holistic, keypoint_map):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image)
        keypoints = preprocess_frames(result, keypoint_map)
        frame_buffer.append(keypoints)

        yield frame  # Yield the frame for display
    cap.release()


