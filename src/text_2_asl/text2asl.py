import os

# Load the word-to-sign mapping from wlasl_class_list.txt
def load_sign_mapping(file_path):
    word_to_sign = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sign_id, word = parts
                    word_to_sign[word.lower()] = int(sign_id)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    return word_to_sign

# Convert ASL phrase to sign IDs
def asl_phrase_to_sign_ids(asl_phrase, word_to_sign):
    tokens = asl_phrase.lower().split()
    sign_ids = [word_to_sign.get(word, -1) for word in tokens if word in word_to_sign]
    if -1 in sign_ids:
        missing_words = [tokens[i] for i, sign_id in enumerate(sign_ids) if sign_id == -1]
        print(f"Warning: Words not found in mapping: {', '.join(missing_words)}")
        sign_ids = [sid for sid in sign_ids if sid != -1]  # Filter out missing words
    return sign_ids

# Save sign IDs to a file
def save_sign_ids_to_file(sign_ids, output_file):
    try:
        with open(output_file, 'w') as file:
            file.write(','.join(map(str, sign_ids)))
        print(f"Sign IDs saved to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")

# Main program
def main():
    # Define file paths
    chatbot_output_file = "chatbot_output.txt"  # File containing ASL-structured text
    mapping_file = "wlasl_class_list.txt"  # Sign mapping file
    output_sign_file = "asl_signs_output.txt"  # Output file for sign IDs

    # Load mapping
    word_to_sign = load_sign_mapping(mapping_file)
    if not word_to_sign:
        return

    # Check if chatbot output exists
    if not os.path.exists(chatbot_output_file):
        print(f"Error: {chatbot_output_file} not found. Make sure chatbot.py outputs to this file.")
        return

    # Read chatbot output
    with open(chatbot_output_file, 'r') as file:
        asl_phrase = file.read().strip()
        print(f"ASL Phrase Read: {asl_phrase}")

    # Convert ASL phrase to sign IDs
    sign_ids = asl_phrase_to_sign_ids(asl_phrase, word_to_sign)
    print(f"Generated Sign IDs: {sign_ids}")

    # Save sign IDs to output file
    save_sign_ids_to_file(sign_ids, output_sign_file)

if __name__ == "__main__":
    main()
