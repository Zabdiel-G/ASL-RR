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
    
def buffer_predictions(predictions, sentence_buffer, end_token="STOP"):
    """
    Adds predictions to the sentence buffer and checks for the end condition.

    Args:
        predictions (list): List of top predictions (label, probability).
        sentence_buffer (list): Buffer to hold sentence words.
        end_token (str): Token that signals the end of a sentence.

    Returns:
        str: Complete sentence if the end condition is met; otherwise, None.
    """
    if not predictions or len(predictions) == 0:
        return None

    # Add the top prediction (most confident) to the sentence buffer
    label, confidence = predictions[0]
    sentence_buffer.append(label)

    # Check if the end token is detected
    if label == end_token:
        # Form the sentence and clear the buffer
        sentence = " ".join(sentence_buffer[:-1])  # Exclude the end token
        sentence_buffer.clear()
        return sentence

    return None
    