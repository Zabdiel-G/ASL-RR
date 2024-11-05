import cv2
import mediapipe as mp
import torch
from utils import preprocess_frame_to_graph, load_class_mapping
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
class_mapping = load_class_mapping("models/wlasl_class_list.txt")

def recognize_gesture(hands_tensor):
    """
    Recognize the gesture from tensor data of hand landmarks using the TGCN model.
    """
    with torch.no_grad():
        outputs = model(hands_tensor)
        _, predicted = torch.max(outputs, 1)
        gesture_label = class_mapping[predicted.item()]
    return gesture_label

# Video capture for gesture recognition
vidcap = cv2.VideoCapture(0)

try:
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            continue

        # Preprocess the frame to extract graph data suitable for TGCN
        hands_tensor = preprocess_frame_to_graph(frame)

        if hands_tensor is not None:
            gesture = recognize_gesture(hands_tensor)
            cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    vidcap.release()
    cv2.destroyAllWindows()
    hands.close()