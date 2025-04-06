import os
import numpy as np
import time
import cv2
from glob import glob
from asl_response.config import MODIFIED_TABLE, CATEGORIES, FRAME_WIDTH, FRAME_HEIGHT, ANIMATION_SPEED
from asl_response.response_utils import load_keypoints_from_json, draw_keypoints


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


def animate_word(word):
    frames = preload_frames(word)
    if not frames:
        return
    for pose, hand_left, hand_right in frames:
        frame = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        draw_keypoints(frame, pose, hand_left, hand_right)
        # cv2.imshow("Sign Language Animator", frame)
        # if cv2.waitKey(ANIMATION_SPEED) & 0xFF == ord('q'):
            # return
        yield frame
        time.sleep(ANIMATION_SPEED/1000)

def animate_sentence(words):
    for word in words:
        for frame in animate_word(word):
            yield frame
        time.sleep(0.3)



