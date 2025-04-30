import threading
import numpy as np
import cv2
import time
from processor import match_sentence_to_words, load_default_pose
from animator import animate_sentence
from response_utils import draw_keypoints
from config import FRAME_WIDTH, FRAME_HEIGHT

current_sentence = None
lock = threading.Lock()

def input_thread():
    global current_sentence
    while True:
        sentence = input("Enter a sentence (or 'quit' to exit): ").strip().lower()
        with lock:
            current_sentence = sentence
        if sentence == "quit":
            break

def response_main():
    global current_sentence
    default_pose, default_hand_left, default_hand_right = load_default_pose()
    threading.Thread(target=input_response, daemon=True).start()

    valid_points = [point for point in default_pose if point is not None]
    if valid_points:
        pose_center_x = sum(p[0] for p in valid_points) / len(valid_points)
        pose_center_y = sum(p[1] for p in valid_points) / len(valid_points)
    else:
        pose_center_x, pose_center_y = 0, 0

    frame_center = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
    offset = (frame_center[0] - pose_center_x, frame_center[1] - pose_center_y)
    scale_factor = 3.0

    while True:
        with lock:
            sentence = current_sentence
            current_sentence = None

        if sentence == ":exit":
            break
        elif sentence:
            words = match_sentence_to_words(sentence)
            print(f"[Matched Words] {words}")
            animate_sentence(words)

        frame = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        # Pass the offset and scale factor to the draw_keypoints function
        draw_keypoints(frame, default_pose, default_hand_left, default_hand_right, offset=offset, scale=scale_factor)
        cv2.imshow("Sign Language Animator", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
