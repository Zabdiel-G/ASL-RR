import cv2
import mediapipe as mp
import numpy as np
import torch

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def preprocess_frame_to_graph(frame):
    """
    Processes a single video frame to extract hand landmarks and convert them into a tensor suitable for TGCN.

    Args:
        frame (np.array): A single frame from a video capture device.

    Returns:
        torch.Tensor: A tensor of hand landmarks suitable for graph neural network input.
    """
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Save memory by making image non-writable

    # Process the image to find hand landmarks
    results = hands.process(image)
    image.flags.writeable = True  # Make image writable again

    # Check if we have any hand landmarks
    if results.multi_hand_landmarks:
        # Here we assume each hand as a graph; you might need to adjust based on your TGCN model input expectations
        all_hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates and normalize them by image dimensions
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            all_hands_data.append(hand_data)

        # Convert the list of hands to a PyTorch tensor
        hands_tensor = torch.tensor(all_hands_data)

        return hands_tensor

    # Return None or an empty tensor if no hands are detected
    return None

def close_mediapipe():
    """
    Release MediaPipe resources. Call this function at the end of your script.
    """
    hands.close()

def load_class_mapping(filepath = "models/wlasl_class_list.txt"):
    """
    Loads the class mapping from a specified text file.
    """
    class_mapping = {}
    with open(filepath, "r") as file:
        for line in file:
            index, label = line.strip().split(maxsplit=1)
            class_mapping[int(index)] = label
    return class_mapping