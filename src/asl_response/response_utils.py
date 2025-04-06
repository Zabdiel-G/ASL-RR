import json
import cv2
from asl_response.config import POSE_PAIRS, HAND_PAIRS

def load_keypoints_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    people = data.get('people', [])
    if not people:
        return [], [], []

    person = people[0]

    def parse_keypoints(flat_list):
        return [
            (flat_list[i], flat_list[i + 1]) if flat_list[i + 2] > 0.1 else None
            for i in range(0, len(flat_list), 3)
        ]

    pose = parse_keypoints(person.get('pose_keypoints_2d', []))
    hand_left = parse_keypoints(person.get('hand_left_keypoints_2d', []))
    hand_right = parse_keypoints(person.get('hand_right_keypoints_2d', []))
    return pose, hand_left, hand_right

def draw_keypoints(frame, pose, hand_left, hand_right):
    for a, b in POSE_PAIRS:
        if a < len(pose) and b < len(pose) and pose[a] and pose[b]:
            cv2.line(frame, tuple(map(int, pose[a])), tuple(map(int, pose[b])), (255, 0, 0), 2)
    for point in pose:
        if point:
            cv2.circle(frame, tuple(map(int, point)), 3, (0, 255, 0), -1)

    for hand, color_line, color_circle in [(hand_left, (0, 0, 255), (0, 255, 255)), (hand_right, (0, 0, 255), (255, 255, 0))]:
        for a, b in HAND_PAIRS:
            if a < len(hand) and b < len(hand) and hand[a] and hand[b]:
                cv2.line(frame, tuple(map(int, hand[a])), tuple(map(int, hand[b])), color_line, 1)
        for point in hand:
            if point:
                cv2.circle(frame, tuple(map(int, point)), 2, color_circle, -1)
    return frame
