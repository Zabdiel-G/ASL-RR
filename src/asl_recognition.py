import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    if thumb_tip.y < index_tip.y:
        return "Thumbs-up"
    return "Unknown gesture"

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

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
                if gesture != "Unknown gesture":
                    speak_text(gesture)

        # Display the frame
        cv2.imshow('ASL Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

vidcap.release()
cv2.destroyAllWindows()