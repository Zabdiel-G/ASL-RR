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

def interpret_asl_input(input_text):
    """
    Converts ASL-structured input into a grammatically correct English sentence.
    """
    prompt = (
        "Convert the following ASL-like phrase into a fluent English sentence. "
        "ASL grammar removes auxiliary verbs, some pronouns, and follows a unique structure. "
        "Your task is to rewrite the ASL phrase in **proper English** with correct grammar and sentence structure.\n\n"

        "Examples:\n\n"

        "**Example 1:**\n"
        "ASL Input: 'how fix computer broken'\n"
        "English Interpretation: 'How can I fix a broken computer?'\n\n"

        "**Example 2:**\n"
        "ASL Input: 'dog bark loud night'\n"
        "English Interpretation: 'The dog barked loudly at night.'\n\n"

        "**Example 3:**\n"
        "ASL Input: 'where find bus stop near'\n"
        "English Interpretation: 'Where can I find the nearest bus stop?'\n\n"

        "**Example 4:**\n"
        "ASL Input: 'friend visit home weekend'\n"
        "English Interpretation: 'My friend is visiting my home this weekend.'\n\n"

        "**Example 5:**\n"
        "ASL Input: 'mother cook dinner family'\n"
        "English Interpretation: 'The mother is cooking dinner for the family.'\n\n"

        "**Now, convert this ASL phrase into proper English:**\n"
        f"ASL Input: {input_text}\n"
        "English Interpretation:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"], 
        max_new_tokens=50, 
        temperature=0.8,  # More flexibility
        top_p=0.95,       # Better response quality
        do_sample=True     # Prevent repeating input text
    )

    english_interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 🔹 **Fix: Ensure output does not contain prompt artifacts**
    # Remove any accidental inclusion of "Example X: ASL Input:"
    if "Example" in english_interpretation:
        english_interpretation = english_interpretation.split("Example")[0].strip()

    return english_interpretation

def generate_english_response(english_text):
    """
    Generates a chatbot response based on a grammatically correct English input.
    """
    prompt = (
        "You are a chatbot. Read the given English sentence and provide a relevant and meaningful response."
        "Focus on creating responses that make sense in English and are conversationally appropriate.\n\n"

        "Example 1:\n"
        "Input: 'What is the capital of France?'\n"
        "Response: 'The capital of France is Paris.'\n\n"
        
        "Example 2:\n"
        "Input: 'How do I cook pasta?'\n"
        "Response: 'Boil water, add salt, cook the pasta for about 8-12 minutes, then drain and serve.'\n\n"
        
        "Example 3:\n"
        "Input: 'Tell me a joke.'\n"
        "Response: 'Why don't skeletons fight each other? Because they don't have the guts!'\n\n"
        
        "Example 4:\n"
        "Input: 'Can you describe how the wind smells?'\n"
        "Response: 'The wind smells fresh, with hints of pine and sea salt, depending on the surroundings.'\n\n"
        
        "Example 5:\n"
        "Input: 'What are some ways to stay healthy?'\n"
        "Response: 'Eating a balanced diet, exercising regularly, getting enough sleep, and managing stress are key ways to stay healthy.'\n\n"

        f"Input: {english_text}\n"
        "Response:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)
    english_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return english_response

# def generate_asl_response(english_text):
def generate_asl_response(english_response):
    """
    Takes an English response and uses a second prompt to rewrite it into a more ASL-like structure,
    then optionally applies simplification rules.
    """
    prompt = (
        "Convert the following English sentence into a simplified ASL-like structure. "
        "Omit auxiliary verbs, pronouns, and unnecessary words. Focus on key concepts and keep the response brief.\n\n"

        "Example 1:\n"
        "English: 'The actor revealed something important and described it.'\n"
        "ASL: 'Actor reveal important describe.'\n\n"

        "Example 2:\n"
        "English: 'Can you help me with this task?'\n"
        "ASL: 'You help me task?'\n\n"

        f"English: {english_response}\n"
        "ASL:"
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
    output_file_path = "chatbot/Text_to_ASL.txt"

    # Check if file exists
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as file:
            inputs = file.readlines()
        
        print("Processing inputs from file:")
        outputs = [] # Stores the results for the output file
        for text in inputs:
            text = text.strip()

            if text:  # Skip empty lines
                print(f"\nInput: {text}")
                
                # Step 1: Convert ASL input to proper English
                english_interpretation = interpret_asl_input(text)
                print("English Input:", english_interpretation)

                # Step 2: Generate a chatbot response
                english_response = generate_english_response(english_interpretation)
                print("English Response:", english_response)

                # Step 3: Convert chatbot response to ASL grammar
                asl_response = generate_asl_response(english_response)
                print("ASL:", asl_response)

                # Append results to output list
                outputs.append(f"Input: {text}\nEnglish Input: {english_interpretation}\nEnglish Response: {english_response}\nASL: {asl_response}\n")
            
        # Write the outputs to the output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n".join(outputs))
        print(f"\nResults written to: {output_file_path}")
    else:
        print(f"File not found: {input_file_path}")

# # Standalone Test for interpret_asl_input()
# if __name__ == "__main__":
#     print("\n Running standalone test for interpret_asl_input():\n")

#     # Test Cases
#     test_cases = [
#         "how fix computer broken",
#         "dog bark loud night",
#         "where find bus stop near",
#         "friend visit home weekend",
#         "mother cook dinner family",
#     ]

#     # Run each test case
#     for test_input in test_cases:
#         result = interpret_asl_input(test_input)
#         print(f"ASL Input: {test_input}")
#         print(f"English Interpretation: {result}\n")

if __name__ == "__main__":
    main()