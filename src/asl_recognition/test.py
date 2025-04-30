import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import cv2
from pathlib import Path

from asl_recognition.pytorch_i3d import InceptionI3d

# Global state
i3d = None
wlasl_dict = {}
sentence_buffer = []

def create_WLASL_dictionary():
    global wlasl_dict
    wlasl_dict = {}
    file_path = Path("asl_recognition/preprocess/wlasl_class_list.txt")
    with open(file_path) as file:
        for line in file:
            split_list = line.strip().split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value

def load_model(weights_path, num_classes):
    global i3d
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights_path))
    i3d = i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    create_WLASL_dictionary()

def run_on_tensor(ip_tensor):
    global sentence_buffer
    ip_tensor = ip_tensor[None, :]  # Add batch dimension
    t = ip_tensor.shape[2]
    ip_tensor = ip_tensor.cuda()

    with torch.no_grad():
        per_frame_logits = i3d(ip_tensor)
        predictions = F.interpolate(per_frame_logits, size=t, mode='linear', align_corners=False)
        predictions = predictions.transpose(2, 1)
        arr = predictions.cpu().detach().numpy()[0]

        probs = F.softmax(torch.from_numpy(arr[0]), dim=0)
        top_prob, top_idx = torch.topk(probs, 1)
        top_prob = top_prob.item()
        top_idx = top_idx.item()

        recognized_word = wlasl_dict[top_idx]
        print(f"[DEBUG] Recognized: {recognized_word} ({top_prob:.2f})")

        if top_prob > 0.10:
            if not sentence_buffer or sentence_buffer[-1] != recognized_word:
                sentence_buffer.append(recognized_word)

        return " ".join(sentence_buffer), recognized_word, top_prob

def recognize_sign(frame, frame_buffer, gesture_sentence, is_recording, is_detecting, start_time, offset):
    global sentence_buffer

    batch = 40
    height, width, _ = frame.shape
    scale_y = 224 / height
    scale_x = 224 / width

    # Preprocess frame
    proc_frame = cv2.resize(frame, dsize=(0, 0), fx=scale_x, fy=scale_y)
    proc_frame = (proc_frame / 255.0) * 2 - 1
    frame_buffer.append(proc_frame)

    if len(frame_buffer) > batch:
        frame_buffer.pop(0)

    state_text = "Capturing" if is_detecting else "Idle"


    if is_recording and len(frame_buffer) == batch:
        tensor_frames = torch.from_numpy(
            np.asarray(frame_buffer, dtype=np.float32).transpose([3, 0, 1, 2])
        )
        run_on_tensor(tensor_frames)
        gesture_sentence = " ".join(sentence_buffer)
        frame_buffer.clear()

    disp_frame = cv2.resize(frame, dsize=(640, 480))
    remaining_time = max(0, int(5 - (time.time() - start_time)))

    return {
        'frame': disp_frame,
        'gesture_sentence': gesture_sentence,
        'state_text': state_text,
        'remaining_time': remaining_time
    }