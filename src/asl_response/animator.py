import os
import numpy as np
import time
import re
import cv2
from glob import glob
from asl_response.config import MODIFIED_TABLE, CATEGORIES, FRAME_WIDTH, FRAME_HEIGHT, ANIMATION_SPEED
from asl_response.response_utils import load_keypoints_from_json, draw_keypoints


def preload_frames(word):
    # 1. Check if it's an explicit fingerspelling pattern like A-B-C
    if re.match(r'^([a-zA-Z]-){1,}[a-zA-Z]$', word):
        letters = word.upper().split('-')
        all_frames = []

        for letter in letters:
            found = False
            for category in CATEGORIES:
                folder_path = os.path.join(MODIFIED_TABLE, category, letter)
                if os.path.exists(folder_path):
                    files = sorted(glob(os.path.join(folder_path, "*.json")))
                    if files:
                        frames = [load_keypoints_from_json(f) for f in files]
                        all_frames.extend(frames)
                        print(f"[Fingerspell] Loaded '{letter}' from {category}/")
                        found = True
                        break
            if not found:
                print(f"[Error] Letter '{letter}' not found.")
        return all_frames

    # 2. Try to load as a regular word
    for category in CATEGORIES:
        folder_path = os.path.join(MODIFIED_TABLE, category, word)
        if os.path.exists(folder_path):
            files = sorted(glob(os.path.join(folder_path, "*.json")))
            if files:
                frames = [load_keypoints_from_json(f) for f in files]
                print(f"[Load] Loaded '{word}' from {category}/")
                return frames
            else:
                print(f"[Error] No frames in '{folder_path}'.")
                return []

    # 3. Fallback: fingerspell each letter of the word
    print(f"[Fallback] Word '{word}' not found. Falling back to fingerspelling.")
    all_frames = []
    for letter in word.upper():
        found = False
        for category in CATEGORIES:
            folder_path = os.path.join(MODIFIED_TABLE, category, letter)
            if os.path.exists(folder_path):
                files = sorted(glob(os.path.join(folder_path, "*.json")))
                if files:
                    frames = [load_keypoints_from_json(f) for f in files]
                    all_frames.extend(frames)
                    print(f"[Fallback-Fingerspell] Loaded '{letter}' from {category}/")
                    found = True
                    break
        if not found:
            print(f"[Error] Letter '{letter}' not found.")
    return all_frames


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



