import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)

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
    (0, 1),  # Nose to Neck
    (1, 2),  # Neck to Right Shoulder
    (2, 3),  # Right Shoulder to Right Elbow
    (3, 4),  # Right Elbow to Right Wrist
    (1, 5),  # Neck to Left Shoulder
    (5, 6),  # Left Shoulder to Left Elbow
    (6, 7),  # Left Elbow to Left Wrist
    (1, 8)   # Neck to Mid Hip
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Estimate Neck and Mid Hip
        neck_x = (landmarks[11].x + landmarks[12].x) / 2
        neck_y = (landmarks[11].y + landmarks[12].y) / 2
        mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

        # Add estimated points to the keypoint_map
        points = {1: (int(neck_x * frame.shape[1]), int(neck_y * frame.shape[0])),
                  8: (int(mid_hip_x * frame.shape[1]), int(mid_hip_y * frame.shape[0]))}

        # Draw keypoints based on mapping and estimated points on the frame
        for mp_index, op_index in keypoint_map.items():
            points[op_index] = (int(landmarks[mp_index].x * frame.shape[1]), int(landmarks[mp_index].y * frame.shape[0]))

        # Draw keypoints
        for point in points.values():
            cv2.circle(frame, point, 5, (255, 0, 0), -1)

        # Draw skeleton
        for start, end in skeleton:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    cv2.imshow('MediaPipe Holistic with Estimated Keypoints and Skeleton', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
