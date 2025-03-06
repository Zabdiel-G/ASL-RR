# Re-import necessary libraries after execution reset
import os
import markovify
import random

# Define the correct file paths dynamically
base_dir = os.getcwd()  # Get current working directory
file_path = os.path.join(base_dir, "models", "filtered_dataset1.txt")  # Input file
cleaned_file_path = os.path.join(base_dir, "models", "predictive_text_cleaned.txt")  # Cleaned output file

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Read and clean the text file
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Remove empty lines and strip extra spaces
cleaned_lines = [line.strip() for line in lines if line.strip()]

# Ensure at least 5 valid lines exist for Markovify
if len(cleaned_lines) < 5:
    raise ValueError(f"Too few valid lines in the file. Found only {len(cleaned_lines)} lines.")

# Save the cleaned text to a new file
with open(cleaned_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

# Read cleaned file for Markovify
with open(cleaned_file_path, "r", encoding="utf-8") as f:
    text = f.read().strip()

# Build the Markovify model
text_model = markovify.NewlineText(text, state_size=1)

# Function to generate the next word using Markovify and a probability distribution
def predict_next_word(last_word, candidate_words_probs):

    try:
        sentence = text_model.make_sentence_with_start(last_word, strict=False)
        if sentence:
            words = sentence.split()
            if len(words) > 1:
                predicted_word = words[1]  # Get the next word
                
                # Check if the predicted word is in the candidate list
                for word, prob in candidate_words_probs:
                    if predicted_word == word:
                        # Return the predicted word
                        return predicted_word

    except (markovify.text.ParamError, KeyError, TypeError):
        print(f"Warning: No valid sentence found for '{last_word}'. Skipping Markov prediction.")

    # If no valid prediction is found, return None
    return None
