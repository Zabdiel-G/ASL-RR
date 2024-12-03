import traceback
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
# from models.TGCN.gen_features import compute_difference

# Initialize MediaPipe Hands
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# keypoint_map = {
#     0: 0,  # Nose
#     5: 15,  # Right eye
#     2: 16,  # Left eye
#     7: 18,  # Left ear
#     8: 17,  # Right ear
#     11: 5,  # Left shoulder
#     12: 2,  # Right shoulder
#     13: 6,  # Left elbow
#     14: 3,  # Right elbow
#     15: 7,  # Left wrist
#     16: 4   # Right wrist
# }

# order of openpose
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

# def preprocess_frames(results):
#     """
#     Extracts keypoints from the results of the MediaPipe Holistic model.
#     """
#     try:
#         # Check if the input is the expected type
#         if not isinstance(results, mp.solutions.holistic.Holistic.__class__):
#             raise ValueError(f"Expected results to be of type SolutionOutputs, got {results}")
        
#         keypoints = []

#         # Check for face landmarks
#         if results.face_landmarks:
#             face_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32)
#             keypoints.append(face_keypoints)

#         # Check for hand landmarks
#         if results.left_hand_landmarks:
#             left_hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
#             keypoints.append(left_hand_keypoints)

#         if results.right_hand_landmarks:
#             right_hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
#             keypoints.append(right_hand_keypoints)

#         # Check for pose landmarks
#         if results.pose_landmarks:
#             pose_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
#             keypoints.append(pose_keypoints)

#         if keypoints:
#             combined_keypoints = np.concatenate(keypoints).flatten()
#             return combined_keypoints
#         return None
#     except Exception as e:
#         print(f"Failed to preprocess frame: {str(e)}")
#         return None
    
def preprocess_frames(results):
    """
    Extracts and combines keypoints from the results of the MediaPipe Holistic model,
    according to the predefined keypoint_map and includes confidence scores.
    """
    try:
        # Check if the input is the expected type
        if not isinstance(results, mp.solutions.holistic.Holistic.__class__):
            raise ValueError(f"Expected results to be of type mp.solutions.holistic.Holistic, got {type(results)}")
        
        # Initialize an array to hold the combined keypoints
        combined_keypoints = []

        # Function to extract keypoints using keypoint_map
        def get_keypoints(landmarks, keypoint_map):
            keypoints = []
            for index, model_index in keypoint_map.items():
                if index < len(landmarks):
                    lm = landmarks[index]
                    keypoints.extend([lm.x, lm.y, lm.visibility])
            return keypoints

        # Extract and combine keypoints based on the mapping
        if results.pose_landmarks:
            pose_keypoints = get_keypoints(results.pose_landmarks.landmark, keypoint_map)
            combined_keypoints.extend(pose_keypoints)
            # print("Pose keypoints:", pose_keypoints)  # Debug print

        # Extract left and right hand keypoints
        if results.left_hand_landmarks:
            left_hand_keypoints = [[lm.x, lm.y, lm.visibility] for lm in results.left_hand_landmarks.landmark]
            combined_keypoints.extend(np.array(left_hand_keypoints).flatten())
            # print("Left hand keypoints:", left_hand_keypoints)  # Debug print

        if results.right_hand_landmarks:
            right_hand_keypoints = [[lm.x, lm.y, lm.visibility] for lm in results.right_hand_landmarks.landmark]
            combined_keypoints.extend(np.array(right_hand_keypoints).flatten())
            # print("Right hand keypoints:", right_hand_keypoints)  # Debug print

        # Print combined keypoints
        print("Combined keypoints:", combined_keypoints)
        return np.array(combined_keypoints)
    except Exception as e:
        print(f"Failed to preprocess frame: {str(e)}")
        return None
    

def compute_difference(keypoints):
    """
    Computes features from keypoints for gesture recognition.
    This function processes keypoints from MediaPipe Holistic, including face, hands, and pose.
    
    Parameters:
    - keypoints: Flattened array of all keypoints from holistic processing.
    
    Returns:
    - Processed features ready for model input.
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    try:
        # Example of computing difference: take difference between each pair of consecutive keypoints
        differences = np.diff(keypoints, axis=0)

        # Flatten the differences to form a feature vector
        feature_vector = differences.flatten()

        # Pad or truncate to fit model's expected input size
        expected_size = 5500  # Update to the correct size based on your model's input layer
        if len(feature_vector) < expected_size:
            feature_vector = np.pad(feature_vector, (0, expected_size - len(feature_vector)), 'constant')
        elif len(feature_vector) > expected_size:
            feature_vector = feature_vector[:expected_size]

        # Assuming model expects shape (55, 100) - update reshape accordingly
        features = feature_vector.reshape((55, 100))
        return features
    except Exception as e:
        print("Error computing differences:", str(e))
        return None

def close_mediapipe():
    """
    Release MediaPipe resources.
    """
    holistic.close()

def load_mapping(filepath = "models/wlasl_class_list.txt"):
    """
    Loads the class mapping from a specified text file.
    """
    class_mapping = {}
    with open(filepath, "r") as file:
        for line in file:
            index, label = line.strip().split(maxsplit=1)
            class_mapping[int(index)] = label
    return class_mapping

class FrameBuffer:
    def __init__(self, maxlen):
        self.frames = deque(maxlen=maxlen)
        
    def add(self, frame):
        self.frames.append(frame)
    
    def is_full(self):
        return len(self.frames) == self.frames.maxlen
    