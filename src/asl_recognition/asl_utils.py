import traceback
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

# Initialize MediaPipe Hands
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def preprocess_frames(results):
    """
    Extracts keypoints from the results of the MediaPipe Holistic model.
    """
    try:
        # Check if the input is the expected type
        if not isinstance(results, mp.solutions.holistic.Holistic.__class__):
            raise ValueError(f"Expected results to be of type SolutionOutputs, got {results}")
        
        keypoints = []

        # Check for face landmarks
        if results.face_landmarks:
            face_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32)
            keypoints.append(face_keypoints)

        # Check for hand landmarks
        if results.left_hand_landmarks:
            left_hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            keypoints.append(left_hand_keypoints)

        if results.right_hand_landmarks:
            right_hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            keypoints.append(right_hand_keypoints)

        # Check for pose landmarks
        if results.pose_landmarks:
            pose_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            keypoints.append(pose_keypoints)

        if keypoints:
            combined_keypoints = np.concatenate(keypoints).flatten()
            return combined_keypoints
        return None
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
    