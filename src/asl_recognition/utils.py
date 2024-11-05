import cv2
import mediapipe as mp
import numpy as np
import torch

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def preprocess_frame(frame):
    """
    Converts frame to a format suitable for gesture recognition.
    """
    try:
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Save memory by making image non-writable

        # Process the image to find hand landmarks
        results = hands.process(image)
        image.flags.writeable = True  # Make image writable again

        if results.multi_hand_landmarks:
            keypoints = [np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                        for hand_landmarks in results.multi_hand_landmarks]
            return keypoints
        return None
    except Exception as e:
        print("Failed to process frame:", str(e))
        traceback.print_exc()

        # Return None or an empty array if processing fails
        return None

def compute_difference(keypoints):
    """
    Computes differences between pairs of keypoints.
    """
    if not keypoints:
        return None

    # Assuming keypoints is a list of (x, y, z) tuples
    # Flatten the keypoints list to a single array and reshape
    keypoints_array = np.array(keypoints).flatten()
    
    # Check if the keypoints have the expected number of elements
    expected_elements = 100 * 256  # Adjust this based on your model's expectation
    if len(keypoints_array) < expected_elements:
        # Pad the keypoints array if it is too short
        keypoints_array = np.pad(keypoints_array, (0, expected_elements - len(keypoints_array)), mode='constant')
    elif len(keypoints_array) > expected_elements:
        # Truncate the keypoints array if it is too long
        keypoints_array = keypoints_array[:expected_elements]

    # Reshape to the shape expected by the model (100, 256)
    features = keypoints_array.reshape((100, 256))

    return features


def close_mediapipe():
    """
    Release MediaPipe resources.
    """
    hands.close()

def load_mapping(filepath = "models/wlasl_class_list.txt"):
    """
    Loads the class mapping from a specified text file.
    """
    class_mapping = {}
    with open(filepath, "r") as file:
        for line in file:
            index, label = line.strip().split(maxsplit=1)
            class_mapping[int(index)] = label
    return class_mapping