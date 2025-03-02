import re
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load GPT-Neo model and tokenizer
model_name = "google/flan-t5-base"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def simplify_to_asl_grammar(text):
    """
    Simplifies English sentences to basic ASL grammar by removing auxiliary verbs, 
    non-essential pronouns, and other unnecessary words.
    """
    # List of auxiliary verbs to remove
    auxiliary_verbs = r'\b(am|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|shall|should|may|might|must|can|could)\b'
    # List of non-essential pronouns
    non_essential_pronouns = r'\b(I|you|we|he|she|it|they)\b'
    
    # Remove auxiliary verbs
    text = re.sub(auxiliary_verbs, '', text, flags=re.IGNORECASE)
    # Remove non-essential pronouns
    text = re.sub(non_essential_pronouns, '', text, flags=re.IGNORECASE)
    # Remove extra spaces caused by deletions
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_english_response(input_text):
    prompt = (
        "You are given the following words or phrases. Your task is to incorporate them into a coherent, meaningful English sentence. "
        "If some words do not fit naturally, feel free to adjust their form slightly but keep their essence. "
        "Focus on creating a sentence that makes sense and conveys a clear idea.\n\n"
        f"Words: {input_text}\n\n"
        "Sentence:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=20, temperature=0.7, top_p=0.9, do_sample=True)
    english_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return english_response

def generate_asl_response(english_text):
    """
    Takes an English response and uses a second prompt to rewrite it into a more ASL-like structure,
    then optionally applies simplification rules.
    """
    # Prompt the model to convert the English text into an ASL-like structure
    prompt = (
        "Rewrite the following English sentence into a simplified ASL-like structure. "
        "Remove auxiliary verbs, pronouns, and unnecessary words. Focus on key words and essential meaning.\n\n"
        f"English: {english_text}\nASL-like Response:"
    )
    
    # Tokenize and generate using the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=20,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode the model's response
    asl_like_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Optionally, run the simplification function again if you want additional refinement
    simplified_response = simplify_to_asl_grammar(asl_like_response)
    
    return simplified_response


def main():
    # File path for input
    input_file_path = "asl_recognition/ASL_to_Text.txt"
    # output_file_path = "chatbot/Text_to_ASL.txt"

    # Check if file exists
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as file:
            inputs = file.readlines()
        
        print("Processing inputs from file:")
        for text in inputs:
            text = text.strip()
            # if text:  # Skip empty lines
            #     print(f"\nInput: {text}")
            #     print("ASL-Structured Response:", generate_asl_response(text))
            if text:  # Skip empty lines
                print(f"\nInput: {text}")
                
                # First, generate the English response
                english_resp = generate_english_response(text)
                print("Pre-simplified (English) response:", english_resp)
                
                # Then convert that English response to ASL grammar
                asl_resp = generate_asl_response(english_resp)
                print("Post-simplified (ASL) response:", asl_resp)
    else:
        print(f"File not found: {input_file_path}")

if __name__ == "__main__":
    main()

