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

def significant_movement(frame_buffer, threshold=0.05):
    if len(frame_buffer) < 2:
        return False 
        
    movement = np.abs(np.diff(frame_buffer, axis=0))
    return np.any(movement > threshold)
    
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
