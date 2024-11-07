from collections import deque
import os
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
from asl_utils import preprocess_frames, compute_difference, close_mediapipe, load_mapping
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
    # print(results)

    # Draw the landmarks on the frame for visualization
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())

    # Pass the 'results' object to the preprocessing function
    keypoints = preprocess_frames(results)

    if keypoints is not None:
        features = compute_difference(keypoints)
        #print("Features (sample):", features[:10])
        if features is not None:
            # Convert features to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Debug: Print initial shape of features_tensor
            # print("Features Tensor Shape:", features_tensor.shape)

            # Ensure correct input shape for the model
            # features_tensor = features_tensor.view(1, 1, 25600)
            # features_tensor = F.adaptive_avg_pool1d(features_tensor, 5500)
            # features_tensor = features_tensor.view(1, 55, 100)

            # Debug: Print final shape of features_tensor to verify correctness
            # print("Final Features Tensor Shape:", features_tensor.shape)

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
