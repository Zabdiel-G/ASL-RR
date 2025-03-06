from predictive_texting import predict_next_word

def log_full_sentence(sentence, log_file="full_sentence.txt"):
    """
    Logs the full sentence to a text file.
    
    Args:
        sentence (str): The full sentence to log.
        log_file (str): File path to store the sentence.
    """
    if not sentence:
        print("No sentence to log.")
        return

    with open(log_file, "a") as f:  # Append to the file
        f.write(sentence + "\n")

    print(f"Full sentence logged to {log_file}")
    
def buffer_predictions(predictions, last_word, candidate_words_probs, end_token="STOP"):
    """
    Args:
        predictions (list): List of top predictions (label, probability).
        last_word (str): The last word from the log file to determine context.
        candidate_words_probs (list): List of (word, probability) pairs.
        end_token (str): Token that signals the end of a sentence.

    Returns:
        str: Complete sentence if the end condition is met; otherwise, None.
    """
    if not predictions or len(predictions) == 0:
        return None

    label, confidence = predictions[0]

    if label == end_token:
        return end_token  

    next_word = predict_next_word(last_word, candidate_words_probs)

    return next_word if next_word else label
    