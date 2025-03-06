import cv2
import mediapipe as mp
import time
import os
import sys
import torch
import numpy as np 
import mediapipe as mp
import random
from models.TGCN.tgcn_model import GCN_muti_att  # Import your model class
from collections import deque
from preprocessing import preprocess_frames, get_keypoints, normalize_keypoints
from asl_utils import load_mapping, keypoint_map, significant_movement, extract_last_word
from gesture_recognition import make_prediction
from asl_to_chatbot import buffer_predictions, log_full_sentence
from predictive_texting import predict_next_word

def process_video_stream(
    model,
    class_mapping: dict[int, str],
    holistic: any,
    frames_to_extract: int = 50,
    sampling_duration: int = 2,
    frame_rate: int = 100,
    resolution_x: int = 720, 
    resolution_y: int = 720,
    movement_threshold: float = 0.01,
    log_file: str = "full_sentence.txt"
):
    """
    Captures video from the default camera, processes each frame to detect gestures,
    and logs recognized gestures into a sentence.

    Args:
        model (torch.nn.Module): The trained gesture recognition model.
        class_mapping (Dict[int, str]): Mapping from class indices to gesture labels.
        holistic (Any): Initialized MediaPipe Holistic model.
        frames_to_extract (int, optional): Number of frames to sample for prediction. Defaults to 50.
        sampling_duration (int, optional): Duration (in seconds) to sample frames. Defaults to 4.
        frame_rate (int, optional): Frame rate for sampling. Defaults to 100.
        resolution_x (int, optional): Horizontal resolution of the video capture. Defaults to 720.
        resolution_y (int, optional): Vertical resolution of the video capture. Defaults to 720.
        movement_threshold (float, optional): Threshold to detect significant movement. Defaults to 0.01.
        log_file (str, optional): File path to log recognized sentences. Defaults to "full_sentence.txt".
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

    frame_buffer: Deque[np.ndarray] = deque(maxlen=frame_rate * sampling_duration)
    sentence_buffer = []
    is_detecting = False
    is_recording = False
    start_time = time.time()
    gesture_sentence = ""

    while cap.isOpened():
        ret, frame = cap.read()
        result = None
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        if is_detecting:
            result = holistic.process(image)
            if result and (result.pose_landmarks or result.left_hand_landmarks or result.right_hand_landmarks):
                _, keypoints = preprocess_frames(result)
            for _ in range(2):  
                frame_buffer.append(keypoints)  

        elapsed_time = time.time() - start_time

        if is_detecting and elapsed_time >= sampling_duration:
            if significant_movement(frame_buffer, threshold=movement_threshold):
                print(f"Current frame buffer length: {len(frame_buffer)}")
                if len(frame_buffer) >= frames_to_extract:
                    try:
                        top_predictions = make_prediction(frame_buffer, model, class_mapping)
                        candidate_words_probs = [(label, confidence) for label, confidence in top_predictions]
                        last_word = sentence_buffer[-1] if sentence_buffer else None

                        print("Top Predictions:")
                        for label, confidence in top_predictions:
                            print(f"{label}: {confidence:.2%}")

                        next_word = buffer_predictions(top_predictions, last_word, candidate_words_probs, end_token="STOP")

                        if next_word is None:
                            next_word = top_predictions[0][0]

                        if next_word:
                            sentence_buffer.append(next_word)
                            print(f"Updated Sentence Buffer: {' '.join(sentence_buffer)}")

                        if next_word == "STOP":
                            complete_sentence = " ".join(sentence_buffer[:-1])  # Remove "STOP" from sentence
                            print(f"Complete Sentence: {complete_sentence}")
                            log_full_sentence(complete_sentence, log_file)
                            sentence_buffer.clear() 
                    except ValueError as e:
                        print(e)
                else:
                    print("Not enough frames to perform prediction.")
            else:
                print("No significant movement detected. Skipping prediction.")

            frame_buffer.clear()
            start_time = time.time()    

        # Display current detection state
        state_text = "STARTING" if is_detecting else "NOT STARTING"
        cv2.putText(
            frame,
            f"State: {state_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if is_detecting else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        remaining_time = max(0, sampling_duration - elapsed_time)
        timer_text = f"Timer: {remaining_time:.1f}s"
        cv2.putText(
            frame,
            timer_text,
            (10, 100),  # Display below the state text
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White color for the timer text
            2,
            cv2.LINE_AA,
        )

        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        with open("ASL_to_Text.txt", "w") as file:
            file.write(gesture_sentence.strip())  # Strip trailing space
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Toggle start/stop
            is_detecting = not is_detecting
            rdingis_reco = not is_recording
        # elif key == ord('r'):
        #     gesture_sentence = ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the trained model
    num_samples = 50  # Ensure this matches the input expected by your model
    input_feature = num_samples * 2
    model = GCN_muti_att(input_feature=input_feature, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
    model_path = os.path.join("models", "archived_TCGN", "asl2000", "ckpt.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    class_mapping = load_mapping("models/wlasl_class_list.txt")

    # Initialize MediaPipe Holistic
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Start the video processing
    process_video_stream(model, class_mapping, holistic)
