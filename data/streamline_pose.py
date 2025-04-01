import os
import json
import shutil

# === CONFIG ===
LOOKUP_JSON_PATH = 'look_up_table.json'
SOURCE_DIR = 'pose_per_individual_videos'
MODIFIED_TABLE = 'modified_table'

# Create modified_table and category folders
CATEGORIES = ['single_word', 'multi_word', 'vowel_word']
for category in CATEGORIES:
    os.makedirs(os.path.join(MODIFIED_TABLE, category), exist_ok=True)

# Load lookup table
with open(LOOKUP_JSON_PATH, 'r') as f:
    lookup_data = json.load(f)

for entry in lookup_data:
    gloss = entry['gloss'].strip()
    pose_ids = entry['pose_ids']
    
    if not pose_ids:
        print(f"[Skip] No pose_ids for word: {gloss}")
        continue

    first_pose_id = pose_ids[0]
    source_pose_folder = os.path.join(SOURCE_DIR, first_pose_id)

    if not os.path.exists(source_pose_folder):
        print(f"[Skip] Source folder not found for pose_id: {first_pose_id}")
        continue

    # Temporary target path to copy JSONs under modified_table/[gloss]/
    temp_word_folder = os.path.join(MODIFIED_TABLE, gloss)
    os.makedirs(temp_word_folder, exist_ok=True)

    # Copy JSON files directly into [gloss] folder (no intermediate folder)
    for filename in os.listdir(source_pose_folder):
        src_file = os.path.join(source_pose_folder, filename)
        dest_file = os.path.join(temp_word_folder, filename)
        shutil.copy2(src_file, dest_file)
    print(f"[Copied] pose_id {first_pose_id} JSONs to '{gloss}'")

# === Organize word folders into categories ===
for word_folder in os.listdir(MODIFIED_TABLE):
    word_path = os.path.join(MODIFIED_TABLE, word_folder)
    if not os.path.isdir(word_path):
        continue  # Skip files

    # Skip category folders themselves
    if word_folder in CATEGORIES:
        continue

    # Determine category
    if len(word_folder) == 1 and word_folder.lower() in 'abcdefghijklmnopqrstuvwxyz':
        category = 'alphabet'
    elif ' ' in word_folder:
        category = 'multi_word'
    else:
        category = 'single_word'

    # Create alphabet category folder if it doesn't exist
    dest_category_path = os.path.join(MODIFIED_TABLE, category)
    os.makedirs(dest_category_path, exist_ok=True)

    # Move [word] folder into appropriate category folder
    dest_path = os.path.join(dest_category_path, word_folder)
    if os.path.exists(dest_path):
        print(f"[Skip] Folder '{dest_path}' already exists.")
        continue

    shutil.move(word_path, dest_path)
    print(f"[Moved] '{word_folder}' → {category}/")
