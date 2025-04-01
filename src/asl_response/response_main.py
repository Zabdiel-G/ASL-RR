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

def main_loop():
    global current_sentence
    default_pose, default_hand_left, default_hand_right = load_default_pose()
    threading.Thread(target=input_thread, daemon=True).start()

    while True:
        with lock:
            sentence = current_sentence
            current_sentence = None

        if sentence == "quit":
            break
        elif sentence:
            words = match_sentence_to_words(sentence)
            print(f"[Matched Words] {words}")
            animate_sentence(words)

        frame = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        draw_keypoints(frame, default_pose, default_hand_left, default_hand_right)
        cv2.imshow("Sign Language Animator", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
