import os
import json
import shutil

# === CONFIG ===
WLASL_JSON_PATH = 'WLASL_v0.3.json'
SOURCE_DIR = 'pose_per_individual_videos'
MODIFIED_TABLE = '../modified_table'

# === Create target structure ===
CATEGORIES = ['single_word', 'multi_word', 'vowel_word']
for category in CATEGORIES:
    os.makedirs(os.path.join(MODIFIED_TABLE, category), exist_ok=True)

# === Load WLASL JSON ===
with open(WLASL_JSON_PATH, 'r') as f:
    wlasl_data = json.load(f)

# === Copy files ===
for entry in wlasl_data:
    gloss = entry['gloss'].strip().lower()
    instances = entry.get('instances', [])

    if not instances:
        print(f"[Skip] No instances for word: {gloss}")
        continue

    # Use the first available instance
    video_id = instances[0].get('video_id')
    if not video_id:
        print(f"[Skip] No video_id for word: {gloss}")
        continue

    source_pose_folder = os.path.join(SOURCE_DIR, video_id)

    if not os.path.exists(source_pose_folder):
        print(f"[Skip] Source folder not found for video_id: {video_id}")
        continue

    # Temporary word folder under modified_table/
    temp_word_folder = os.path.join(MODIFIED_TABLE, gloss)
    os.makedirs(temp_word_folder, exist_ok=True)

    for filename in os.listdir(source_pose_folder):
        src_file = os.path.join(source_pose_folder, filename)
        dest_file = os.path.join(temp_word_folder, filename)
        shutil.copy2(src_file, dest_file)

    print(f"[Copied] video_id {video_id} JSONs to '{gloss}'")

# === Organize words into categories ===
for word_folder in os.listdir(MODIFIED_TABLE):
    word_path = os.path.join(MODIFIED_TABLE, word_folder)
    if not os.path.isdir(word_path):
        continue

    if word_folder in CATEGORIES:
        continue

    if len(word_folder) == 1 and word_folder.lower() in 'abcdefghijklmnopqrstuvwxyz':
        category = 'alphabet'
    elif ' ' in word_folder:
        category = 'multi_word'
    else:
        category = 'single_word'

    dest_category_path = os.path.join(MODIFIED_TABLE, category)
    os.makedirs(dest_category_path, exist_ok=True)

    dest_path = os.path.join(dest_category_path, word_folder)
    if os.path.exists(dest_path):
        print(f"[Skip] Folder '{dest_path}' already exists.")
        continue

    shutil.move(word_path, dest_path)
    print(f"[Moved] '{word_folder}' → {category}/")
