import cv2
import mediapipe as mp
import torch
from utils import preprocess_frame, compute_difference, close_mediapipe, load_mapping
from models.TGCN.tgcn_model import GCN_muti_att

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load the trained TGCN model
model_path = 'models/archived_TCGN/asl2000/ckpt.pth'
model = GCN_muti_att(input_feature=100, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Load class mapping
class_mapping = load_mapping("models/wlasl_class_list.txt")

# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to get it into the correct format
    keypoints = preprocess_frame(frame)
    if keypoints is not None:
        features = compute_difference(keypoints)  # Compute differences or any other necessary transformations
        if features is not None:
            # Convert features to tensor and add a batch dimension
            features_tensor = torch.tensor([features], dtype=torch.float32)

            # Make predictions
            with torch.no_grad():
                outputs = model(features_tensor)
                predicted_gesture = torch.argmax(outputs, dim=1)
                print(f"Predicted Gesture: {predicted_gesture.item()}")

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_mediapipe()