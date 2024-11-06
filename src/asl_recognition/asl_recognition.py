import os
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from collections import deque
from utils import preprocess_frame, compute_difference, close_mediapipe, load_mapping
from models.TGCN.tgcn_model import GCN_muti_att

# Ensure TensorFlow and any CUDA-based libraries do not attempt to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Initialize MediaPipe Hands with CPU settings (no GPU dependency)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

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

# Use a deque to keep track of the last 10 predictions for a sliding window effect
predictions_window = deque(maxlen=144)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to get it into the correct format
    keypoints = preprocess_frame(frame)
    if keypoints is not None:
        features = compute_difference(keypoints)  # Compute differences or other transformations
        if features is not None:
            # Convert features to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape [1, 25600]

            # Downsample to match model's expected input shape [1, 55, 100]
            features_tensor = features_tensor.view(1, 1, 25600)  # Add channel dimension for pooling
            features_tensor = F.adaptive_avg_pool1d(features_tensor, 5500)  # Downsample to 5500 elements
            features_tensor = features_tensor.view(1, 55, 100)  # Reshape to [1, 55, 100]

            # Make prediction
            with torch.no_grad():
                outputs = model(features_tensor)
                predicted_gesture = torch.argmax(outputs, dim=1)
                gesture_name = class_mapping.get(predicted_gesture.item(), "Unknown Gesture")
                
                # Add the prediction to the sliding window
                predictions_window.append(gesture_name)
                
                # Compute the most frequent gesture in the sliding window
                most_frequent_gesture = max(set(predictions_window), key=predictions_window.count)
                print(f"Predicted Gesture (smoothed): {most_frequent_gesture}")

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_mediapipe()

