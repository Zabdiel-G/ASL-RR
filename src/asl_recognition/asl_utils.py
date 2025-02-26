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

    # Function to get the last word from the log file
def extract_last_word(log_file):
    """
    Reads the last word from the log file.
    If the file is empty, returns None.
    """
    if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
        return None  # Log file is empty, return None

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return None  # No lines in file

    last_line = lines[-1].strip()
    words = last_line.split()

    return words[-1] if words else None
