import os
import numpy as np

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODIFIED_TABLE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Data", "modified_table"))

print(f"[Debug] MODIFIED_TABLE Path: {MODIFIED_TABLE}")

# Words are split into three types to help with 
CATEGORIES = ['multi_word', 'single_word', 'alphabet']

#This is for how fast the pose will sign.
ANIMATION_SPEED = 10

POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Preload Multi-Word Folder Names
multi_word_path = os.path.join(MODIFIED_TABLE, 'multi_word')
multi_word_folders = set(os.listdir(multi_word_path)) if os.path.exists(multi_word_path) else set()

# Config for the image resolution 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BLANK_FRAME = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

