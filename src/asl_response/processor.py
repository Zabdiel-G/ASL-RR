import os
from glob import glob
from asl_response.config import MODIFIED_TABLE, CATEGORIES, multi_word_folders
from asl_response.response_utils import load_keypoints_from_json

#breaks down sentence into words and handles 
#compound word such as "a lot", "United State"
#Todo: maybe add a fingerspelling mechanism
def match_sentence_to_words(sentence):
    words = sentence.lower().split()
    matched = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            phrase = f"{words[i]} {words[i+1]}"
            if phrase in multi_word_folders:
                matched.append(phrase)
                i += 2
                continue
        matched.append(words[i])
        i += 1
    return matched

def load_default_pose():
    for category in CATEGORIES:
        path = os.path.join(MODIFIED_TABLE, category)
        if not os.path.exists(path):
            print(f"[Skip] Missing category: {path}")
            continue
        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue
            json_files = sorted(glob(os.path.join(folder_path, "*.json")))
            if json_files:
                print(f"[Info] Default pose from: {json_files[0]}")
                return load_keypoints_from_json(json_files[0])
    raise ValueError("No valid pose found.")
