from collections import deque
import os
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
from asl_utils import preprocess_frames, compute_difference, close_mediapipe, load_mapping, keypoint_map, skeleton
from models.TGCN.tgcn_model import GCN_muti_att

# Ensure TensorFlow and any CUDA-based libraries do not attempt to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Initialize MediaPipe Hands with CPU settings (no GPU dependency)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained TGCN model with original input feature size
model = GCN_muti_att(input_feature=100, hidden_feature=256,
                     num_class=2000, p_dropout=0.3, num_stage=24)
model_path = os.path.join("models", "archived_TCGN", "asl2000", "ckpt.pth")
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Load class mapping
class_mapping = load_mapping("models/wlasl_class_list.txt")

# Initialize camera (OpenCV uses CPU by default)
cap = cv2.VideoCapture(0)

# Use a deque to store logits of the last N frames for averaging
logits_window = deque(maxlen=144)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Estimate Neck and Mid Hip
        neck_x = (landmarks[11].x + landmarks[12].x) / 2
        neck_y = (landmarks[11].y + landmarks[12].y) / 2
        mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

        # Add estimated points to the keypoint_map
        points = {1: (int(neck_x * image.shape[1]), int(neck_y * image.shape[0])),
                  8: (int(mid_hip_x * image.shape[1]), int(mid_hip_y * image.shape[0]))}

        # Draw keypoints based on mapping and estimated points on the frame
        for mp_index, op_index in keypoint_map.items():
            points[op_index] = (int(landmarks[mp_index].x * image.shape[1]), int(landmarks[mp_index].y * image.shape[0]))

        # Draw keypoints
        for point in points.values():
            cv2.circle(image, point, 5, (255, 0, 0), -1)

        # Draw skeleton
        for start, end in skeleton:
            if start in points and end in points:
                cv2.line(image, points[start], points[end], (0, 0, 255), 2)

    # Pass the 'results' object to the preprocessing function
    keypoints = preprocess_frames(results)

    if keypoints is not None:
        features = compute_difference(keypoints)
        #print("Features (sample):", features[:10])
        if features is not None:
            # Convert features to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = model(features_tensor)
                logits_window.append(outputs)
                avg_logits = torch.mean(torch.stack(list(logits_window)), dim=0)
                predicted_gesture = torch.argmax(avg_logits, dim=1)
                gesture_name = class_mapping.get(predicted_gesture.item(), "Unknown Gesture")
                print(f"Predicted Gesture: {gesture_name}")
                cv2.putText(image, f'Gesture: {gesture_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('ASL Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_mediapipe()
