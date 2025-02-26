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
    """
    Uses Markovify to predict the next word and selects from candidate words with their probabilities.
    If no valid word is found, falls back to a weighted random selection.
    Lists an odd number of top words and their probability.
    """
    predicted_word = None
    try:
        sentence = text_model.make_sentence_with_start(last_word, strict=False)
        if sentence:
            words = sentence.split()
            if len(words) > 1:
                predicted_word = words[1]  # Get the next word
                
                # If the predicted word is in the candidate list, return it
                for word, prob in candidate_words_probs:
                    if predicted_word == word:
                        sorted_predictions = sorted(candidate_words_probs, key=lambda x: x[1], reverse=True)
                        return predicted_word, sorted_predictions

    except (markovify.text.ParamError, KeyError):
        print(f"Warning: No valid sentence found for '{last_word}'. Choosing based on probability distribution.")

    # Fallback: Weighted random selection based on candidate probabilities
    words, probabilities = zip(*candidate_words_probs)  # Unpack words and their probabilities
    chosen_word = random.choices(words, weights=probabilities, k=1)[0]

    # Return the chosen word and top candidate words sorted by probability
    return chosen_word, sorted(candidate_words_probs, key=lambda x: x[1], reverse=True)

# Example Usage: Predict next word with an odd number of candidate words
last_word = "woman"

# Ensure the number of candidate words is odd and use non-round probabilities
candidate_words_probs = [("bed", 0.37), ("receive", 0.23), ("believe", 0.11), ("die", 0.19), ("voice", 0.10)]

# Run prediction
next_word, top_predictions = predict_next_word(last_word, candidate_words_probs)

# Display results
print("Predicted Next Word:", next_word)
print("\nTop Predictions:")
for word, prob in top_predictions:
    print(f"{word}: {prob:.2%}")  # Display as a percentage
