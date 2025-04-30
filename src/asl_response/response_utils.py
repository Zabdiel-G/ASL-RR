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

def draw_keypoints(frame, pose, hand_left, hand_right, offset=(0, 0), scale=1.0, pivot=None):
    offset_x, offset_y = offset

    # Determine the pivot if not provided: use center of the pose keypoints
    if pivot is None:
        valid_points = [point for point in pose if point is not None]
        if valid_points:
            pivot_x = sum(p[0] for p in valid_points) / len(valid_points)
            pivot_y = sum(p[1] for p in valid_points) / len(valid_points)
            pivot = (pivot_x, pivot_y)
        else:
            pivot = (0, 0)

    def scale_point(point):
        # Scale the point relative to the pivot, then apply the offset.
        new_x = pivot[0] + scale * (point[0] - pivot[0]) + offset_x
        new_y = pivot[1] + scale * (point[1] - pivot[1]) + offset_y
        return (int(new_x), int(new_y))

    for a, b in POSE_PAIRS:
        if a < len(pose) and b < len(pose) and pose[a] and pose[b]:
            pt_a = scale_point(pose[a])
            pt_b = scale_point(pose[b])
            cv2.line(frame, pt_a, pt_b, (255, 0, 0), 2)
    for point in pose:
        if point:
            pt = scale_point(point)
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

    for hand, color_line, color_circle in [
        (hand_left, (0, 0, 255), (0, 255, 255)),
        (hand_right, (0, 0, 255), (255, 255, 0))
    ]:
        for a, b in HAND_PAIRS:
            if a < len(hand) and b < len(hand) and hand[a] and hand[b]:
                pt_a = scale_point(hand[a])
                pt_b = scale_point(hand[b])
                cv2.line(frame, pt_a, pt_b, color_line, 1)
        for point in hand:
            if point:
                pt = scale_point(point)
                cv2.circle(frame, pt, 2, color_circle, -1)
    return frame
