import cv2
import torch
import torch.nn.functional as F
from models.pytorch_i3d import InceptionI3d
import numpy as np

# Load the ASL model
#new
model = InceptionI3d(num_classes=2000)  # Set for ASL2000
model.load_state_dict(torch.load("models/archived/asl2000/ASL2000.pt", map_location=torch.device('cpu')))
model.eval()

# Load the class mapping
class_mapping = {}
with open("models/wlasl_class_list.txt", "r") as file:
    for line in file:
        index, label = line.strip().split(maxsplit=1)
        class_mapping[int(index)] = label

# Initialize camera
vidcap = cv2.VideoCapture(0)

# Settings
frame_buffer = []
frame_count = 0  # Counter to skip frames
skip_frames = 5  # Process every 5th frame for smoother response

print("Starting ASL recognition...")

while vidcap.isOpened():
    ret, frame = vidcap.read()
    if not ret:
        break

    frame_count += 1

    # Resize and process frame
    resized_frame = cv2.resize(frame, (299, 299))  # Resize to model input size
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.tensor(rgb_frame).float().permute(2, 0, 1) / 255.0  # Normalize and permute dimensions

    # Append frame to buffer and maintain length of 8 frames
    frame_buffer.append(frame_tensor)
    if len(frame_buffer) > 8:
        frame_buffer.pop(0)

    # Run inference every 5 frames if we have 8 frames in buffer
    if frame_count % skip_frames == 0 and len(frame_buffer) == 8:
        video_input = torch.stack(frame_buffer, dim=1).unsqueeze(0)  # Shape: [1, 3, 8, 299, 299]
        
        with torch.no_grad():
            prediction = model(video_input)  # Run model inference
            probabilities = F.softmax(prediction, dim=1)  # Apply softmax for confidence
            max_prob, predicted_class_index = torch.max(probabilities, dim=1)  # Get max probability and class
            max_prob = max_prob.item()

        # Confidence threshold to filter low-confidence predictions
        if max_prob > 0.7:
            gesture_label = class_mapping.get(predicted_class_index.item(), "Unknown gesture")
        else:
            gesture_label = "No gesture detected"

        # Print predicted gesture
        print("Predicted gesture:", gesture_label)

        # Display prediction on the frame
        cv2.putText(frame, f'Gesture: {gesture_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

# Release resources
vidcap.release()
cv2.destroyAllWindows()

