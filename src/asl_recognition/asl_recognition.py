import os
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
from utils import preprocess_frame, compute_difference, close_mediapipe, load_mapping
from models.TGCN.tgcn_model import GCN_muti_att

# Ensure TensorFlow and any CUDA-based libraries do not attempt to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Initialize MediaPipe Hands with CPU settings (no GPU dependency)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.6)

# Load the trained TGCN model with original input feature size
model = GCN_muti_att(input_feature=100, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
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
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to get it into the correct format
    keypoints = preprocess_frame(frame)
    if keypoints is not None:
        features = compute_difference(keypoints)
        print("Features (sample):", features[:10])   
        if features is not None:
            # Convert features to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0) 

            # Ensure correct input shape for the model
            features_tensor = features_tensor.view(1, 1, 25600)  
            features_tensor = F.adaptive_avg_pool1d(features_tensor, 5500)  
            features_tensor = features_tensor.view(1, 55, 100)  
            
            with torch.no_grad():
                outputs = model(features_tensor)
                logits_window.append(outputs)  
                avg_logits = torch.mean(torch.stack(list(logits_window)), dim=0)
                predicted_gesture = torch.argmax(avg_logits, dim=1)
                gesture_name = class_mapping.get(predicted_gesture.item(), "Unknown Gesture")
                print(f"Predicted Gesture (smoothed): {gesture_name}")

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_mediapipe()





