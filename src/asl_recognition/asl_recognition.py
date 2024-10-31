#imported libraries

import cv2
import mediapipe as mp
import torch
from pytorch_i3d import InceptionI3d
#import pyttsx3

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the WLASL pre-trained I3D model
model = InceptionI3d(num_classes=200)  # Set based on WLASL
model.load_state_dict(torch.load("models/archived/asl300/ASL300.pt", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Define a mapping for model output to gesture labels
class_mapping = {0: "Hello", 1: "Thank you", 2: "Yes", ...}  # Map as per WLASL classes

def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the thumb is pointing upwards
    is_thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < index_tip.y

    thumb_angle = abs(thumb_ip.x - thumb_tip.x)
    index_angle = abs(index_tip.x - thumb_ip.x)

    if is_thumb_up and thumb_angle < 0.1 and index_angle < 0.1:
        return "Thumbs-up"
    return "Unknown gesture"

#def speak_text(text):
#    engine.say(text)
#    engine.runAndWait()

# Start capturing video from the webcam
vidcap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            print("Ignoring empty frame")
            continue

        # Convert the image from BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe to detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                if gesture != "Unknown gesture":
#                    speak_text(gesture)

        # Display the frame
        cv2.imshow('ASL Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

vidcap.release()
cv2.destroyAllWindows()