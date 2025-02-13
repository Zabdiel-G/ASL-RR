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
from asl_utils import load_mapping, keypoint_map, significant_movement
from gesture_recognition import make_prediction
from asl_to_chatbot import buffer_predictions, log_full_sentence

def process_video_stream(
    model,
    class_mapping: dict[int, str],
    holistic: any,
    frames_to_extract: int = 50,
    sampling_duration: int = 4,
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
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = holistic.process(image)

        # Preprocess the frame to extract keypoints
        _, keypoints = preprocess_frames(result)
        frame_buffer.append(keypoints)
        elapsed_time = time.time() - start_time

        if is_detecting and elapsed_time >= sampling_duration:
            if significant_movement(frame_buffer, threshold=movement_threshold):
                print(f"Current frame buffer length: {len(frame_buffer)}")
                if len(frame_buffer) >= frames_to_extract:
                    try:
                        top_predictions = make_prediction(frame_buffer, model, class_mapping)
                        # Print the top predictions
                        print("Top Predictions:")
                        for label, confidence in top_predictions:
                            print(f"{label}: {confidence:.2f}")

                        # Process the predictions as needed
                        sentence = buffer_predictions(top_predictions, sentence_buffer, end_token="STOP")
                        if sentence:
                            print(f"Complete Sentence: {sentence}")
                            log_full_sentence(sentence, log_file)
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Toggle start/stop
            is_detecting = not is_detecting
            # Append "STOP" to the sentence buffer when space is pressed
            sentence_buffer.append("STOP")
            sentence = " ".join(sentence_buffer[:-1])  # Form the sentence excluding the "STOP" token
            if sentence:
                print(f"Complete Sentence: {sentence}")
                log_full_sentence(sentence, log_file)
        elif key == ord('q'):  # Quit
            break

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
