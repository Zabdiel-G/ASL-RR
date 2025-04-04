import os
import numpy as np
import time
import cv2
from glob import glob
from config import MODIFIED_TABLE, CATEGORIES, FRAME_WIDTH, FRAME_HEIGHT, ANIMATION_SPEED
from response_utils import load_keypoints_from_json, draw_keypoints


def preload_frames(word):
    for category in CATEGORIES:
        folder_path = os.path.join(MODIFIED_TABLE, category, word)
        if os.path.exists(folder_path):
            files = sorted(glob(os.path.join(folder_path, "*.json")))
            if not files:
                print(f"[Error] No frames in '{folder_path}'.")
                return []
            frames = [load_keypoints_from_json(f) for f in files]
            print(f"[Load] Loaded '{word}' from {category}/")
            return frames
    print(f"[Error] Word '{word}' not found.")
    return []


def animate_word(word, scale=3.0):
    frames = preload_frames(word)
    if not frames:
        return
    for pose, hand_left, hand_right in frames:
        frame = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        valid_points = [point for point in pose if point is not None]
        if valid_points:
            pose_center_x = sum(p[0] for p in valid_points) / len(valid_points)
            pose_center_y = sum(p[1] for p in valid_points) / len(valid_points)
        else:
            pose_center_x, pose_center_y = 0, 0
        
        frame_center = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
        
        offset = (frame_center[0] - pose_center_x, frame_center[1] - pose_center_y)
        
        draw_keypoints(frame, pose, hand_left, hand_right, offset=offset, scale=scale)
        cv2.imshow("Sign Language Animator", frame)
        if cv2.waitKey(ANIMATION_SPEED) & 0xFF == ord('q'):
            return


def animate_sentence(words):
    for word in words:
        animate_word(word)
        time.sleep(0.3)



