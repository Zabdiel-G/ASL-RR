import cv2
import mediapipe as mp
import os
import numpy as np
import torch
from collections import deque
from models.TGCN.tgcn_model import GCN_muti_att  # Import your model class
from asl_utils import preprocess_frames, compute_difference, close_mediapipe, load_mapping, keypoint_map, skeleton
import time
import random


# Load the trained model
num_samples = 50 # Ensure this matches the input expected by your model
input_feature = num_samples * 2
model = GCN_muti_att(input_feature=input_feature, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
model_path = os.path.join("models", "archived_TCGN", "asl2000", "ckpt.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
class_mapping = load_mapping("models/wlasl_class_list.txt")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)


# Keypoint mapping and skeleton for visualization
keypoint_map = {
    0: 0,  # Nose
    5: 15,  # Right eye
    2: 16,  # Left eye
    7: 18,  # Left ear
    8: 17,  # Right ear
    11: 5,  # Left shoulder
    12: 2,  # Right shoulder
    13: 6,  # Left elbow
    14: 3,  # Right elbow
    15: 7,  # Left wrist
    16: 4   # Right wrist
}

skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), 
    (0, 16), (0, 15), (16, 18), (15, 17)
]

# Prediction history
logits_window = deque(maxlen=144)  # Use a sliding window to smooth predictions

        # Process pose landmarks
def get_keypoints(landmarks, keypoint_map):
    """
    Extracts and scales keypoints (x, y) from the provided landmarks based on a keypoint mapping.
    Scales the normalized keypoints to the image resolution.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    keypoints = []
    for index, model_index in keypoint_map.items():
        if index < len(landmarks):
            lm = landmarks[index]
            keypoints.extend([lm.x, lm.y])
        else:
            # Pad with zeros if the landmark is missing
            keypoints.extend([0.0, 0.0])
    return np.array(keypoints, dtype=np.float32).reshape(-1, 2)

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints[:, 0] = 2 * ((keypoints[:, 0] * image_width) / image_width - 0.5)  # Normalize x
    keypoints[:, 1] = 2 * ((keypoints[:, 1] * image_height) / image_height - 0.5)  # Normalize y
    return keypoints


# Preprocess Frames Function
def preprocess_frames(results):
    """
    Extracts keypoints and calculates specific points from the results of the MediaPipe Holistic model.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    try:
        points = {}
        keypoints = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate Neck and Mid Hip
            neck_x = (landmarks[11].x + landmarks[12].x) / 2
            neck_y = (landmarks[11].y + landmarks[12].y) / 2
            mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

            pose_keypoints = get_keypoints(landmarks, keypoint_map)

            # Add neck and mid-hip to the pose keypoints
            neck_point = [neck_x, neck_y ]
            mid_hip_point = [mid_hip_x , mid_hip_y]
            pose_keypoints = np.insert(pose_keypoints, [1, 8], [neck_point, mid_hip_point], axis=0)

            # Normalize keypoints to [-1, 1]
            # pose_keypoints = 2 * ((torch.FloatTensor(pose_keypoints) / torch.FloatTensor([256, 256])) - 0.5)
            pose_keypoints = normalize_keypoints(pose_keypoints, image_width=256, image_height=256)


            # Save points for visualization
            for idx, (x, y) in enumerate(pose_keypoints):
                points[idx] = (int(x), int(y))

            keypoints.append(pose_keypoints)
            #print("pose landmarks")
        else:
            # Pad with zeros if pose landmarks are missing
            keypoints.append(np.zeros((13, 2), dtype=np.float32))

        # Process left hand landmarks
        if results.left_hand_landmarks:
            left_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            left_hand_keypoints = normalize_keypoints(left_hand_keypoints, image_width=256, image_height=256)
            keypoints.append(left_hand_keypoints)
            #print("left hand landmarks")

        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))        
    
        # Process right hand landmarks
        if results.right_hand_landmarks:
            right_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            right_hand_keypoints = normalize_keypoints(right_hand_keypoints, image_width=256, image_height=256)  # Apply normalization
            keypoints.append(right_hand_keypoints)
            #print("right hand landmarks")
        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))
            
        combined_keypoints = np.concatenate(keypoints).flatten() if keypoints else None
        
        return points, combined_keypoints
    except Exception as e:
        print(f"Failed to preprocess frame: {str(e)}")
        return {}, None
    
    
def significant_movement(frame_buffer, threshold=0.01):
    if len(frame_buffer) < 2:
        return False 
        
    movement = np.abs(np.diff(frame_buffer, axis=0))
    return np.any(movement > threshold)

# Parameters
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

frame_rate = 100
sampling_duration = 4
frames_to_extract = 50  # Number of frames to sample
frame_buffer = deque(maxlen=frame_rate * sampling_duration)
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
        if significant_movement(frame_buffer, threshold=0.01):
            print(f"Current frame buffer length: {len(frame_buffer)}")
            if len(frame_buffer) >= frames_to_extract:
                sampled_frames = random.sample(list(frame_buffer), frames_to_extract)

                input_data = np.stack(sampled_frames).T
                input_data = input_data.reshape(55, 100)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)
                    top_probs = top_probs[0].cpu().numpy()
                    top_classes = top_classes[0].cpu().numpy()

                    top_predictions = [
                        (class_mapping.get(cls, "Unknown"), prob)
                        for cls, prob in zip(top_classes, top_probs)
                    ]

                # Print the top predictions
                print("Top Predictions:")
                for label, confidence in top_predictions:
                    print(f"{label}: {confidence:.2f}")
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

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord(' '):  # Toggle start/stop
        is_detecting = not is_detecting

cap.release()
cv2.destroyAllWindows()
