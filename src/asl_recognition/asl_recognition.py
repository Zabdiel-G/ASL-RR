import cv2
import mediapipe as mp
import torch
from models.TGCN.tgcn_model import GCN
from utils import preprocess_frame_to_graph, close_mediapipe

# Load the TGCN model
model = load_tgcn_model('models/TGCN/tgcn_model.py')
model.eval()  # Set the model to evaluation mode

# Load class mapping
class_mapping = load_class_mapping("models/wlasl_class_list.txt")

def recognize_gesture(hands_tensor):
    """
    Recognize the gesture from tensor data of hand landmarks using the TGCN model.

    Args:
        hands_tensor (torch.Tensor): The input tensor for the TGCN model containing hand landmarks data.

    Returns:
        str: The recognized gesture label.
    """
    with torch.no_grad():
        output = model(hands_tensor)
        gesture_class = output.argmax(dim=1).item()
        gesture_label = class_mapping.get(gesture_class, "Unknown gesture")
    return gesture_label

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Start capturing video from the webcam
vidcap = cv2.VideoCapture(0)

try:
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Preprocess the frame to extract graph data suitable for TGCN
        hands_tensor = preprocess_frame_to_graph(frame)

        if hands_tensor is not None:
            # Recognize gestures
            gesture = recognize_gesture(hands_tensor)
            cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

finally:
    vidcap.release()
    cv2.destroyAllWindows()
    close_mediapipe()  # Clean up MediaPipe resources