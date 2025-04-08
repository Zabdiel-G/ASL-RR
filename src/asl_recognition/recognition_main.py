import math
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
from pathlib import Path

import numpy as np
import cv2

i3d = None
wlasl_dict = {}
sentence_buffer = []

def create_WLASL_dictionary():
    global wlasl_dict 
    wlasl_dict = {}
    file_path = Path("preprocess" ) / "wlasl_class_list.txt"

    with file_path.open() as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
def send_sentence(sentence):
    # have no clue how to send it rn. 
    # For now I made it similiar to the previous iteration
    print("Sending sentence:", sentence)

def run_on_tensor(ip_tensor):
    #also took the base from WLASL
    global sentence_buffer
    ip_tensor = ip_tensor[None, :]  
    t = ip_tensor.shape[2]
    ip_tensor = ip_tensor.cuda()
    
    per_frame_logits = i3d(ip_tensor)
    predictions = F.upsample(per_frame_logits, t, mode='linear')
    predictions = predictions.transpose(2, 1)
    arr = predictions.cpu().detach().numpy()[0]
    
    probs = F.softmax(torch.from_numpy(arr[0]), dim=0)
    top_prob, top_idx = torch.topk(probs, 1)
    top_prob = top_prob.item()
    top_idx = top_idx.item()
    
    recognized_word = wlasl_dict[top_idx]
    output_str = f"{recognized_word}: {top_prob:.2f}"
    print(output_str)
    
    if top_prob > 0.20:
        if not sentence_buffer or sentence_buffer[-1] != recognized_word:
            sentence_buffer.append(recognized_word)
    
    return " ".join(sentence_buffer)

def recognition_main_frontend():
#Took from WLASL test_i3D for the how it the data should be
#The frames were scaled to 224, but most resolution should be compatible
#https://github.com/dxli94/WLASL/blob/master/code/I3D/test_i3d.py
    global sentence_buffer
    vidcap = cv2.VideoCapture(0)
    frames = []
    offset = 0
    text = ""
    batch = 40
    font = cv2.FONT_HERSHEY_TRIPLEX  
    word_count = 0
    
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        
        offset += 1
        
        height, width, _ = frame.shape
        scale_y = 224 / height
        scale_x = 224 / width
        
        proc_frame = cv2.resize(frame, dsize=(0, 0), fx=scale_x, fy=scale_y)
        proc_frame = (proc_frame / 255.0) * 2 - 1
        
        disp_frame = cv2.resize(frame, dsize=(1280, 720))
        
        frames.append(proc_frame)
        if len(frames) > batch:
            frames.pop(0)
        
        # I got the idea of running inference every 20 frames from as well as the model in the link
        #https://github.com/alanjeremiah/WLASL-Recognition-and-Translation
        if len(frames) == batch and (offset == batch or offset % 20 == 0):
            tensor_frames = torch.from_numpy(
                np.asarray(frames, dtype=np.float32).transpose([3, 0, 1, 2])
            )
            text = run_on_tensor(tensor_frames)
        
        cv2.putText(disp_frame, text, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
        cv2.imshow('frame', disp_frame)
        
        result_dict = {
            'gesture_sentence': " ".join(sentence_buffer),
        }
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        yield result_dict
        
    vidcap.release()
    cv2.destroyAllWindows()
    
    if sentence_buffer:
        final_sentence = " ".join(sentence_buffer)
        send_sentence(final_sentence)
        sentence_buffer.clear()

def load_model(weights, num_classes):
    global i3d 
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    
    for result in recognition_main_frontend():
        print("Frontend update:", {k: result[k] for k in result if k != 'frame'})

if __name__ == '__main__':
    create_WLASL_dictionary()
    weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    num_classes = 2000
    load_model(weights, num_classes)
