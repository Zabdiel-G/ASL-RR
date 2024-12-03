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

# Preprocess Frames Function
def preprocess_frames(results):
    """
    Extracts keypoints and calculates specific points from the results of the MediaPipe Holistic model.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    try:
        points = {}
        keypoints = []

        # Process pose landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate Neck and Mid Hip
            neck_x = (landmarks[11].x + landmarks[12].x) / 2
            neck_y = (landmarks[11].y + landmarks[12].y) / 2
            mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

            # Include calculated points
            desired_pose_indices = [0, 5, 2, 7, 8, 11, 12, 13, 14, 15, 16]  # Include OpenPose indices
            pose_keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in desired_pose_indices], dtype=np.float32)
            neck_point = [neck_x, neck_y]
            mid_hip_point = [mid_hip_x, mid_hip_y]
            pose_keypoints = np.insert(pose_keypoints, [1, 8], [neck_point, mid_hip_point], axis=0)

            for idx, (x, y) in enumerate(pose_keypoints):
                points[idx] = (int(x * 256), int(y * 256))

            keypoints.append(pose_keypoints)
        else:
            keypoints.append(np.zeros((13, 2), dtype=np.float32))

        # Process left hand landmarks
        if results.left_hand_landmarks:
            left_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            keypoints.append(left_hand_keypoints)
        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))

        # Process right hand landmarks
        if results.right_hand_landmarks:
            right_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            keypoints.append(right_hand_keypoints)
        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))

        # Combine all keypoints into a single array
        combined_keypoints = np.concatenate(keypoints).flatten() if keypoints else None
        # if combined_keypoints is not None:
        #     print("Combined Keypoints Array Shape:", combined_keypoints.shape)
        #     print("Combined Keypoints Array:", combined_keypoints)
        # else:
        #     print("Combined Keypoints is None.")
        
        return points, combined_keypoints
    except Exception as e:
        print(f"Failed to preprocess frame: {str(e)}")
        return {}, None

# Parameters
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

num_frames = 50  # Number of frames required for the model
num_keypoints = 55  # Number of keypoints expected by the model
frame_rate = 10  # Frames per second to sample
sampling_duration = 5 # Duration to collect frames (in seconds)
frame_buffer = deque(maxlen=frame_rate * sampling_duration) 



start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Convert the frame to RGB (required by MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = holistic.process(image)


    # Preprocess the frame to extract keypoints
    _,keypoints = preprocess_frames(result)
    frame_buffer.append(keypoints)

    # Check if enough time has passed to sample frames
    elapsed_time = time.time() - start_time
    if elapsed_time >= sampling_duration:
        # Randomly sample 50 frames from the buffer
        if len(frame_buffer) >= num_frames:
            sampled_frames = random.sample(list(frame_buffer), num_frames)

            # Create input for the model (55 keypoints × 100 values)
            input_data = np.stack(sampled_frames).T  # Transpose to shape [keypoints, frames * 2]
            input_data = input_data.reshape(55, 100)    # Reshape to (55, 100)

            # Convert to a PyTorch tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Perform prediction
            with torch.no_grad():
                outputs = model(input_tensor)  # Forward pass
                predicted_class = torch.argmax(outputs, dim=1).item()  # Get predicted class index
                predicted_label = class_mapping.get(predicted_class, "Unknown")
                
                # Print the prediction
                print(f"Predicted Gesture: {predicted_label}")

        # Reset buffer and timer for the next round
        frame_buffer.clear()
        start_time = time.time()

    # Display the frame
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
