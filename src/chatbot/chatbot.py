import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

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

def generate_asl_response(input_text):
    """
    Generates a response using FLAN-T5 and simplifies it to basic ASL grammar.
    """    
    if input_text.strip().endswith(("what", "where", "how","?")):
        response_type = "Answer briefly in ASL grammar."
    else:
        response_type = "Continue the conversation naturally in ASL grammar. Respond briefly using essential words only."
    
    # Define the prompt for FLAN-T5
    prompt = (
        "{response_type}\n\n"
        "Examples:\n"
        "Input: 'Your name what?'\nResponse: 'My name flan.'\n\n"
        "Input: 'You help me, this?'\nResponse: 'Yes, help you.'\n\n"
        f"Input: '{input_text}'\nResponse:"
    )
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate the response
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=20,  # Limit to keep response concise
        temperature=0.4,  # Lower for more straightforward responses
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode and simplify the response
    asl_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    simplified_response = simplify_to_asl_grammar(asl_response)
    
    return asl_response

# Test with additional inputs
examples = [
    "DAY GOOD, YOU?",
    "hello",
    "Weather today what?",
    "You help me, this?",
    "trains station where?",
    "Me Student.",
    "Today weather nice, yes?",
]

# File path for input
file_path = "../asl_recognition/ASL_to_Text.txt"

# Check if file exists
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        inputs = file.readlines()
    
    print("Processing inputs from file:")
    for text in inputs:
        text = text.strip()
        if text:  # Skip empty lines
            print(f"\nInput: {text}")
            print("ASL-Structured Response:", generate_asl_response(text))
else:
    print(f"File not found: {file_path}")
