import cv2
import mediapipe as mp
import os
import numpy as np
import torch
from collections import deque
from asl_recognition.models.TGCN.tgcn_model import GCN_muti_att  # Import your model class
from asl_recognition.asl_utils import preprocess_frames, compute_difference, close_mediapipe, load_mapping, keypoint_map, skeleton
import time
import random


# Load the trained model
num_samples = 50 # Ensure this matches the input expected by your model
frames_to_extract = 50 
sampling_duration = 4
input_feature = num_samples * 2
model = GCN_muti_att(input_feature=input_feature, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
model_path = os.path.join("asl_recognition", "models", "archived_TCGN", "asl2000", "ckpt.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
class_mapping = load_mapping("asl_recognition/models/wlasl_class_list.txt")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Keypoint mapping and skeleton for visualization
keypoint_map = {
    0: 0,  # Nose
    5: 15,  # Right eye
    2: 16,  # Left eye
    7: 18,  # Left ear
    8: 17,  # Right ear
    11: 5,  # Left shoulder
    12: 2,  # Right shoulder
    13: 6,  # Left elbow
    14: 3,  # Right elbow
    15: 7,  # Left wrist
    16: 4   # Right wrist
}

skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), 
    (0, 16), (0, 15), (16, 18), (15, 17)
]

# Prediction history
logits_window = deque(maxlen=144)  # Use a sliding window to smooth predictions

        # Process pose landmarks
def get_keypoints(landmarks, keypoint_map):
    """
    Extracts and scales keypoints (x, y) from the provided landmarks based on a keypoint mapping.
    Scales the normalized keypoints to the image resolution.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    keypoints = []
    for index, model_index in keypoint_map.items():
        if index < len(landmarks):
            lm = landmarks[index]
            keypoints.extend([lm.x, lm.y])
        else:
            # Pad with zeros if the landmark is missing
            keypoints.extend([0.0, 0.0])
    return np.array(keypoints, dtype=np.float32).reshape(-1, 2)

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints[:, 0] = 2 * ((keypoints[:, 0] * image_width) / image_width - 0.5)  # Normalize x
    keypoints[:, 1] = 2 * ((keypoints[:, 1] * image_height) / image_height - 0.5)  # Normalize y
    return keypoints


# Preprocess Frames Function
def preprocess_frames(results):
    """
    Extracts keypoints and calculates specific points from the results of the MediaPipe Holistic model.
    Pads missing keypoints with zeros to ensure consistent output size.
    """
    try:
        points = {}
        keypoints = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate Neck and Mid Hip
            neck_x = (landmarks[11].x + landmarks[12].x) / 2
            neck_y = (landmarks[11].y + landmarks[12].y) / 2
            mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

            pose_keypoints = get_keypoints(landmarks, keypoint_map)

            # Add neck and mid-hip to the pose keypoints
            neck_point = [neck_x, neck_y ]
            mid_hip_point = [mid_hip_x , mid_hip_y]
            pose_keypoints = np.insert(pose_keypoints, [1, 8], [neck_point, mid_hip_point], axis=0)

            # Normalize keypoints to [-1, 1]
            # pose_keypoints = 2 * ((torch.FloatTensor(pose_keypoints) / torch.FloatTensor([256, 256])) - 0.5)
            pose_keypoints = normalize_keypoints(pose_keypoints, image_width=256, image_height=256)


            # Save points for visualization
            for idx, (x, y) in enumerate(pose_keypoints):
                points[idx] = (int(x), int(y))

            keypoints.append(pose_keypoints)
            #print("pose landmarks")
        else:
            # Pad with zeros if pose landmarks are missing
            keypoints.append(np.zeros((13, 2), dtype=np.float32))

        # Process left hand landmarks
        if results.left_hand_landmarks:
            left_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            left_hand_keypoints = normalize_keypoints(left_hand_keypoints, image_width=256, image_height=256)
            keypoints.append(left_hand_keypoints)
            #print("left hand landmarks")

        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))        
    
        # Process right hand landmarks
        if results.right_hand_landmarks:
            right_hand_keypoints = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            right_hand_keypoints = normalize_keypoints(right_hand_keypoints, image_width=256, image_height=256)  # Apply normalization
            keypoints.append(right_hand_keypoints)
            #print("right hand landmarks")
        else:
            keypoints.append(np.zeros((21, 2), dtype=np.float32))
            
        combined_keypoints = np.concatenate(keypoints).flatten() if keypoints else None
        
        return points, combined_keypoints
    except Exception as e:
        print(f"Failed to preprocess frame: {str(e)}")
        return {}, None
    
    
def significant_movement(frame_buffer, threshold=0.01):
    if len(frame_buffer) < 2:
        return False 
        
    movement = np.abs(np.diff(frame_buffer, axis=0))
    print("Movement differences:", movement)  # Log movement differences
    return np.any(movement > threshold)


def recognize_sign(frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time):
    """
    Processes one frame, updates the frame buffer, and performs inference if conditions are met.
    Returns a dictionary for frontend integration.
    """
    # Process the frame with MediaPipe
    image = frame.copy()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = holistic.process(image)
    
    # Preprocess the frame and update the buffer
    _, keypoints = preprocess_frames(result)
    if keypoints is not None:
        frame_buffer.append(keypoints)
    
    # Compute elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare common return variables
    state_text = "Capturing" if is_detecting else "Idle"
    remaining_time = max(0, sampling_duration - elapsed_time)
    word_count_text = f"Detected Words: {word_count}"
    
    # Check if we should perform prediction
    if is_detecting and elapsed_time >= sampling_duration:
        # Proceed only if we have enough frames
        if len(frame_buffer) >= frames_to_extract:
            if significant_movement(frame_buffer, threshold=0.01):
                print(f"Current frame buffer length: {len(frame_buffer)}")
                # Sample frames and prepare model input
                sampled_frames = random.sample(list(frame_buffer), frames_to_extract)
                input_data = np.stack(sampled_frames).T
                input_data = input_data.reshape(55, 100)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)
                    top_probs = top_probs[0].cpu().numpy()
                    top_classes = top_classes[0].cpu().numpy()
                    
                    top_predictions = [
                        (class_mapping.get(cls, "Unknown"), prob)
                        for cls, prob in zip(top_classes, top_probs)
                    ]
                    
                    top_label = top_predictions[0][0]
                    if is_recording:
                        gesture_sentence += top_label + " "
                
                word_count += 1
                
                print("Top Predictions:")
                for label, confidence in top_predictions:
                    print(f"{label}: {confidence:.2f}")
                print(f"Word: {word_count}\n")
                
                # Clear the buffer and reset the timer after a prediction
                frame_buffer.clear()
                start_time = time.time()
            else:
                # Print the warning once if the buffer is exactly full
                if len(frame_buffer) == frames_to_extract:
                    print("No significant movement detected. Skipping prediction.")
                # Keep accumulating frames until significant movement is detected.
        else:
            # Optionally, you could log that the buffer isn't full yet:
            # print(f"Buffer length ({len(frame_buffer)}) is less than required ({frames_to_extract}).")
            pass

    # Handle key events if needed (this might be irrelevant in a web context)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Toggle detection state
        is_detecting = not is_detecting
        is_recording = not is_recording
    elif key == ord('r'):
        gesture_sentence = ""
    
    # Return using your current structure
    return {
        'frame': frame,  # or process into base64 for frontend
        'gesture_sentence': gesture_sentence,
        'word_count': word_count,
        'state_text': state_text,
        'remaining_time': remaining_time,
        'word_count_text': word_count_text
    }





# def main():
#     # Parameters
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     frame_rate = 100
#     sampling_duration = 4
#     frames_to_extract = 50  # Number of frames to sample
#     frame_buffer = deque(maxlen=frame_rate * sampling_duration)
#     # String to accumulate recognized gesture names
#     gesture_sentence = ""
#     is_recording = False
#     is_detecting = False

#     start_time = time.time()
#     word_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = frame.copy()
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Process the frame with MediaPipe
#         result = holistic.process(image)

#         # Preprocess the frame to extract keypoints
#         _, keypoints = preprocess_frames(result)
#         frame_buffer.append(keypoints)

#         if result.pose_landmarks:
#             landmarks = result.pose_landmarks.landmark

#             # Estimate Neck and Mid Hip
#             neck_x = (landmarks[11].x + landmarks[12].x) / 2
#             neck_y = (landmarks[11].y + landmarks[12].y) / 2
#             mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
#             mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

#             # Add estimated points to the keypoint_map
#             points = {1: (int(neck_x * frame.shape[1]), int(neck_y * frame.shape[0])),
#                     8: (int(mid_hip_x * frame.shape[1]), int(mid_hip_y * frame.shape[0]))}

#             # Draw keypoints based on mapping and estimated points on the frame
#             for mp_index, op_index in keypoint_map.items():
#                 points[op_index] = (int(landmarks[mp_index].x * frame.shape[1]), int(landmarks[mp_index].y * frame.shape[0]))

#             # Draw keypoints
#             for point in points.values():
#                 cv2.circle(frame, point, 5, (255, 0, 0), -1)

#             # Draw skeleton
#             for start, end in skeleton:
#                 if start in points and end in points:
#                     cv2.line(frame, points[start], points[end], (255, 0, 255), 2)

#         elapsed_time = time.time() - start_time

#         if is_detecting and elapsed_time >= sampling_duration:
#             if significant_movement(frame_buffer, threshold=0.01):
#                 print(f"Current frame buffer length: {len(frame_buffer)}")
#                 if len(frame_buffer) >= frames_to_extract:
#                     sampled_frames = random.sample(list(frame_buffer), frames_to_extract)

#                     input_data = np.stack(sampled_frames).T
#                     input_data = input_data.reshape(55, 100)
#                     input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

#                     with torch.no_grad():
#                         outputs = model(input_tensor)
#                         probabilities = torch.softmax(outputs, dim=1)
#                         top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)
#                         top_probs = top_probs[0].cpu().numpy()
#                         top_classes = top_classes[0].cpu().numpy()

#                         top_predictions = [
#                             (class_mapping.get(cls, "Unknown"), prob)
#                             for cls, prob in zip(top_classes, top_probs)
#                         ]

#                         # Print the top predictions
#                         top_label = top_predictions[0][0]

#                         if is_recording:
#                             gesture_sentence += top_label + " "  

#                     word_count += 1

#                     # Print the top predictions
#                     print("Top Predictions:")
#                     for label, confidence in top_predictions:
#                         print(f"{label}: {confidence:.2f}")
                    
#                     print(f"Word: {word_count}")
#                     print("")
#                 else:
#                     print("Not enough frames to perform prediction.")
#             else:
#                 print("No significant movement detected. Skipping prediction.")

#             frame_buffer.clear()
#             start_time = time.time()

#         # Display current detection state
#         state_text = "Capturing" if is_detecting else "Idle"
#         cv2.putText(
#             frame,
#             f"State: {state_text}",
#             (10, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0) if is_detecting else (0, 0, 255),
#             2,
#             cv2.LINE_AA,
#         )
#         remaining_time = max(0, sampling_duration - elapsed_time)
#         timer_text = f"Timer: {remaining_time:.1f}s"
        
#         cv2.putText(
#             frame,
#             timer_text,
#             (10, 100),  # Display below the state text
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),  # White color for the timer text
#             2,
#             cv2.LINE_AA,
#         )

#         # Display word count
#         word_count_text = f"Detected Words: {word_count}"
#         cv2.putText(
#             frame,
#             word_count_text,
#             (10, 150),  # Position below the timer
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 0),  # White color for text
#             2,
#             cv2.LINE_AA,
#         )

#         # Display the frame
#         cv2.imshow("Webcam Feed", frame)

#         with open("asl_recognition/ASL_to_Text.txt", "w") as file:
#             file.write(gesture_sentence.strip())  # Strip trailing space
#         # Handle key presses
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):  # Quit
#             break
#         elif key == ord(' '):  # Toggle start/stop
#             is_detecting = not is_detecting
#             is_recording = not is_recording
#         elif key == ord('r'):
#             gesture_sentence = ""


#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
