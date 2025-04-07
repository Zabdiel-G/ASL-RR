import re
import os
import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
openai.api_key = os.getenv("OPENAI_API_KEY")

# # Load GPT-Neo model and tokenizer
# # model_name = "google/flan-t5-base"  
# model_name = "google/flan-t5-xl"  
# # model_name = "facebook/bart-large"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load WLASL word list
def load_wlasl_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip().split('\t')[1].lower() for line in f if '\t' in line)

wlasl_words = load_wlasl_words("asl_recognition/preprocess/wlasl_class_list.txt")

# Post-process gloss output to match WLASL list or convert to fingerspelling
def postprocess_asl_gloss(gloss, wlasl_words):
    words = re.findall(r'\b\w+\b', gloss.lower())  # Remove punctuation and lowercase
    processed = []
    for word in words:
        if word in wlasl_words:
            processed.append(word)
        else:
            fingerspelled = '-'.join(list(word))
            processed.append(fingerspelled)
    return ' '.join(processed)

def interpret_asl_input(input_text):
    """
    Converts ASL-structured input into a grammatically correct English sentence.
    """
    prompt = (
        "Convert the following ASL phrase into fluent English while preserving ASL grammar. "
        "ASL follows a topic-comment structure, places time indicators first, and often omits prepositions. "
        "Your task is to rewrite the ASL phrase into correct English without changing its intended meaning.\n\n"

        "ASL Structure Includes:\n"
        "- Time markers first (e.g., 'Yesterday, store I go')\n"
        "- Topic before comment (e.g., 'Pizza, I like')\n"
        "- WH-words at the end (e.g., 'Bus stop where?')\n"
        "- Minimal use of prepositions (e.g., 'Table, cup sit' instead of 'The cup is on the table')\n"
        "- Use of classifier verbs (e.g., 'Vehicle-drive-uphill' instead of 'The car is driving uphill')\n\n"

        "Examples:\n\n"

        "**Example 1:**\n"
        "ASL Input: 'YESTERDAY STORE GO I'\n"
        "English Interpretation: 'Yesterday, I went to the store.'\n\n"

        "**Example 2:**\n"
        "ASL Input: 'BOOK READ FINISH, I UNDERSTAND'\n"
        "English Interpretation: 'After finishing the book, I understood.'\n\n"

        "**Example 3:**\n"
        "ASL Input: 'BUS STOP WHERE?'\n"
        "English Interpretation: 'Where is the bus stop?'\n\n"

        "**Example 4:**\n"
        "ASL Input: 'MY FRIEND HOME VISIT WEEKEND, FRIEND'\n"
        "English Interpretation: 'My friend visited my home this weekend.'\n\n"

        "**Example 5:**\n"
        "ASL Input: 'CUP TABLE SIT'\n"
        "English Interpretation: 'The cup is on the table.'\n\n"

        "**Example 6:**\n"
        "ASL Input: 'DOG CHASE CAT FAST'\n"
        "English Interpretation: 'The dog is chasing the cat quickly.'\n\n"

        "**Now, convert this ASL phrase into proper English:**\n"
        f"ASL Input: {input_text}\n"
        "English Interpretation:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in translating ASL gloss into fluent English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )

        english_interpretation = response['choices'][0]['message']['content'].strip()

        # Optional cleanup (if needed)
        if "Example" in english_interpretation:
            english_interpretation = english_interpretation.split("Example")[0].strip()

        return english_interpretation

    except Exception as e:
        return f"[OpenAI API Error: {e}]"

def generate_english_response(english_text):
    """
    Generates a chatbot response based on a grammatically correct English input.
    """
    prompt = (
        "You are a chatbot. Read the given English sentence and provide a relevant and meaningful response."
        "Focus on creating responses that make sense in English and are conversationally appropriate.\n\n"

        "Do not copy or reuse any of the provided examples. Generate a new response.\n\n"

        "Important: Do not include any example text in your answer.\n\n"

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

        "Important: Do not include any example text in your answer.\n\n"
        "Now, respond to the sentence with a brief response. :\n\n"
        "Keep the response short.\n\n"

        f"Input: {english_text}\n"
        "Response:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly and knowledgeable chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100  # Adjust if needed
        )

        english_response = response['choices'][0]['message']['content'].strip()
        return english_response

    except Exception as e:
        return f"[OpenAI API Error: {e}]"

# def generate_asl_response(english_text):
def generate_asl_response(english_response):
    """
    Takes an English response and uses a second prompt to rewrite it into a more ASL-like structure,
    then optionally applies simplification rules.
    """
    prompt = (
        "You are an expert ASL interpreter. Convert the following English sentence into a single ASL sentence using gloss notation, "
        "applying correct ASL syntax and grammar rules explained below. Do not copy examples; produce a new, fluent ASL sentence. "
        "Your task is to rewrite the ASL phrase into correct English without changing its intended meaning. "
        "Do not copy or reuse any of the provided examples. Generate a new response.\n\n"

        "Important: Do not include any example text in your answer.\n\n"

        "### ASL Linguistic Rules ###\n\n"

        # 1. Time Markers First

        "#### 1. Time Markers First ####\n"
        "**Mnemonic Anchor: 'When first, then what.'**\n"
        "In ASL, adverbs of time (e.g., 'yesterday,' 'tomorrow,' 'last week') typically appear at the beginning of a sentence. "
        "This provides a temporal framework before describing the action. Unlike English, ASL does not use conjugated verbs for tense such as '-ed' to indicate past tense. "
        "Instead, it uses separate lexical signs (e.g., 'yesterday,' 'next year').\n\n"
        "**Why ASL Uses Time Markers First:** \n"
        " - It Establishes Temporal Context: By stating when something happens first, the signer provides a clear timeline before describing the event.\n"
        " - It Replaces Verb Conjugation: Since ASL does not conjugate verbs for tense (like '-ed' in English), time markers function as explicit indicators of time.\n"
        " - It is Similar to Conditional Phrases: Just as conditional phrases ('If it rains...') appear first in ASL, time markers also set up the sentence context early.\n"
        " - It Contributes to Discourse Structure: Organizing time early in a sentence helps clarify event sequences, making conversations more structured.\n\n"
        "While the default placement of time markers is at the beginning, ASL does allow some flexibility. "
        "For emphasis, a signer may move the time marker to a different position, but this is less common. "
        "The preferred and most natural ASL sentence structure **keeps time markers first**.\n\n"
        "Summary: Always begin with time indicators when present (e.g., yesterday, every morning).\n\n"
        "Do not reuse the examples below — they are for structure reference only.\n\n"

        "### Examples of Time Markers First ###\n"

        " - English: 'I went to the store yesterday.'\n"
        "  -> ASL: 'Yesterday store I go'\n\n"
        
        " - English: 'He will travel to Japan next year.'\n"
        "  -> ASL: 'Next year Japan he travel'\n\n"
        
        " - English: 'We met last night at the café.'\n"
        "  -> ASL: 'Last night café we meet'\n\n"
        
        " - English: 'She studies every morning before work.'\n"
        "  -> ASL: 'Every morning work before she study'\n\n"
        # add 4 more "Time Markers First" examples here
        " - English: 'I will have a baby in the near future.'\n"
        "  -> ASL: 'Soon, I have baby'\n\n"

        " - English: 'My 29th birthday is tomorrow!'\n"
        "  -> ASL: 'Tommorow my birthday'\n\n"

        " - English: 'I watched a boring movie last night.'\n"
        "  -> ASL: 'Last night boring movie I watch'\n\n"

        " - English: 'The heat is awful today.'\n"
        "  -> ASL: 'today heat awful'\n\n"

        "\n\n"

        # 2. Topicalization

        "#### 2. Topic Before Comment (Topicalization) ####\n"
        "**Mnemonic Anchor: 'Topic sets the frame. Comment completes the thought.'**\n"
        "ASL often begins with the topic of the sentence, followed by a comment that describes or provides information about that topic. "
        "• The **topic** is the subject or object being discussed—the primary focus of the sentence.\n"
        "• The **comment** is the statement that provides information about the topic.\n"
        "This structure is known as **topicalization**, a key syntactic feature of ASL. "
        "This structure deviates from standard English SVO order to allow emphasis and clarity.\n"

        "**Why ASL Uses 'Topic Before Comment':** \n"
        " - Establishes Information Flow: The topic is introduced **first** to set the context before adding details.\n"
        " - Nonmanual Signals (NMS) for Topicalization:**\n"
        "  * Raised eyebrows\n"
        "  * Head tilt\n"
        "  * Slight pause after the topic\n"
        "  * The topic is often glossed with a lowercase 't' above the sign\n\n"
        "**Relationship to Other ASL Grammar Rules:** \n"
        " - Time Markers First:** Just like ASL places time indicators first (e.g., 'Yesterday, store I go'), "
        "topicalization follows a similar principle by setting up context first.\n"
        " - Flexible Word Order for Emphasis:** Unlike English, which sticks to SVO (Subject-Verb-Object), "
        "ASL allows Topic-Comment structures that deviate from SVO to create emphasis.\n\n"

        "**Key Differences from English:** \n"
        " - English relies on word order (SVO) to clarify meaning.\n"
        " - ASL **does not require subjects to come first**—instead, it prioritizes **what is most important or emphasized**.\n"
        " - Pronouns can be omitted when the topic is clear.**\n\n"

        "**Flexibility of Topicalization:** \n"
        " - While topics usually appear first, ASL can modify sentence structure **for emphasis** or **to reference previous topics.**\n"
        " - Example:\n"
        "  - English: 'The test was hard. I failed it.'\n"
        "  -> ASL: 'Test hard I fail'\n\n"

        "Summary: Introduce the main subject of the sentence first, then provide the comment.\n\n"
        "Do not reuse the examples below. They are for reference only. \n\n"

        "### Examples ###\n"
        " - English: 'I really hate homework.'\n"
        "  -> ASL: 'Homework I detest' (Topic: 'Homework', Comment: 'I detest it.')\n\n"

        " - English: 'I love pizza.'\n"
        "  → ASL: 'Pizza I love'\n\n"
        
        " - English: 'The child, his father loves him.'\n"
        "  → ASL: 'Child father love'\n\n"

        " - English: 'That movie was amazing!'\n"
        "  → ASL: 'Movie amazing'\n\n"

        " - English: 'My dog is very energetic.'\n"
        "  → ASL: 'My dog energetic'\n\n"

        # add 4 more topicalization examples here

        "\n\n"

        # 3. WH -Words

        "#### 3. WH-Words at the End ####\n"
        "**Mnemonic Anchor: 'Say the thing, then ask about it.'**\n"
        "In English, WH-questions (who, what, where, when, why) are generally placed at the beginning of a sentence (e.g., 'Where is the man?'). "
        "In ASL, WH-questions (who, what, where, when, why) are generally placed at the end of a sentence (e.g., 'Man where?'). "
        "These questions are accompanied by nonmanual markers like furrowed eyebrows and a head tilt.\n\n"

        "**Nonmanual Signals (NMS) for WH-Questions:** \n"
        "When signing a WH-question, specific **nonmanual markers** are used to indicate that a question is being asked. These include:\n"
        " - Eyebrows squinted**\n"
        " - Head tilted slightly forward**\n"
        " - Body leaning forward slightly**\n"
        " - Shoulders possibly raised**\n"
        "These nonmanual signals help **differentiate WH-questions from statements** in ASL.\n\n"

        "**Relationship to Other ASL Grammar Rules:** \n"
        " *Time Markers First:** Just like time indicators ('Yesterday, store I go') set up temporal context,"
        " placing WH-words at the end ensures clarity in question formation.\n"
        " *Topic-Comment Structure:** ASL places the **topic first** before asking a WH-question about it. "
        "For example (Do not reuse. Example is just for reference):\n"
        "  - English: 'Where is the bus stop?'\n"
        "  - ASL: 'Bus stop where?'\n\n"

        " **Key Differences from English:** \n"
        " - English uses auxiliary verbs like 'is' (e.g., 'Where is he?')."
        " ASL omits these and moves the WH-word to the end (e.g., 'He where?').\n"
        " - English relies on word order, while ASL uses **nonmanual markers** to indicate a question.\n\n"

        "Summary: Place WH-words at the end of the question. Use proper facial expressions.\n\n"
        "Do not reuse the examples below. They are only for reference. \n\n"

        "### Examples ###\n"

        " - **English:** 'Where is the man?'\n"
        "  → **ASL:** 'Man where? (eyebrows squinted)'\n\n"

        " - **English:** 'Who is your teacher?'\n"
        "  → **ASL:** 'Your teacher who?'\n\n"

        " - **English:** 'What is your favorite color?'\n"
        "  → **ASL:** 'Your favorite color what?'\n\n"

        " - **English:** 'When is the meeting?'\n"
        "  → **ASL:** 'Meeting when?'\n\n"

        " - **English:** 'Why are you late?'\n"
        "  → **ASL:** 'You late why?'\n\n"

        # add 5 more WH-question examples here

        "\n\n"

        # 4. Emphasis through repetition

        "#### 4. Emphasis Through Repetition ####\n"
        "**Mnemonic Anchor: 'Repeat to emphasize.'**\n"
        "In ASL, repetition is frequently used as a strategy to add emphasis, modify meaning, or highlight key information. "
        "ASL uses repetition for emphasis, emotion, or indicating ongoing/repeated action. "
        "This occurs across morphological, lexical, prosodic, discourse, and spatial levels.\n\n"
        "Summary: Repeat signs or modify them to emphasize intensity, frequency, or emotion.\n\n"
        "Do not reuse the examples below. They are just for reference. \n\n"

        " ** Morphological: LOOK++, ASK++ (indicating repetition)\n"
        "  * ASL uses reduplication (repeating part or all of a sign) to emphasize how an action unfolds over time.\n"
        "  * Habitual actions are repeated with shortened movements to show ongoing repetition (e.g., 'LOOK++' for 'keeps looking').\n"
        "  * Iterative actions use repeated movements in a different pattern to show repeated occurrences (e.g., 'ASK++' for 'asked multiple times').\n"
        "Do not reuse the examples below. They are just for reference. \n\n"
        " **Examples:**\n"
        "  - English: 'I always watch TV at night.'\n"
        "  - ASL: 'Night, TV watch++ (habitual reduplication).'\n"
        "  - English: 'He asked me over and over.'\n"
        "  - ASL: 'He ask++ me (iterative reduplication).'\n\n"

        " ** Lexical: HAPPY HAPPY (for very happy)\n"
        "  * Repeating a word (lexeme) in ASL can emphasize its **importance, intensity, or certainty**.\n"
        "Do not reuse the examples below. They are just for reference. \n\n"
        "  * Example: 'HAPPY HAPPY' instead of 'VERY HAPPY' to reinforce emotional intensity.\n"
        " **Examples:**\n"
        "  - English: 'I am really happy!'\n"
        "  - ASL: 'HAPPY HAPPY!'\n"
        "  - English: 'The room was very dark.'\n"
        "  - ASL: 'DARK DARK (intensification).'\n\n"

        " ** Prosodic: Repetition in storytelling or emotion-driven expressions\n"

        " ** Discourse: Repetition for Meaning Amplification \n"
        "  * Signers repeat certain words in narratives to emphasize their importance and **evoke emotion**.\n"
        "  * Example from the Oklahoma City Bombing narrative: The sign 'CRY+++' (cry repeated) reinforced the emotional impact.\n"
        "  * Storytelling repetition keeps **attention and highlights key events**.\n"
        "Do not reuse the examples below. They are just for reference. \n\n"
        " **Examples:**\n"
        "  - English: 'Everyone was crying.'\n"
        "  - ASL: 'CRY CRY CRY+++ (emotional emphasis).'\n\n"

        " ** Spatial: Repetition in storytelling or emotion-driven expressions\n\n"
        "  * In ASL, signers use **repeated references in signing space** to emphasize concepts and maintain discourse flow.\n"
        "  *Example:** Repeatedly pointing to the same location in space to emphasize a particular person or event.\n"
        "Do not reuse the examples below. They are just for reference. \n\n"
        " **Examples:**\n"
        "  - English: 'He kept coming back.'\n"
        "  - ASL: 'HE++ return (spatial repetition).'\n\n"

        " ** Lexicalized: Lexicalized Fingerspelling with Repetition\n"
        "  * Some fingerspelled words that have become ASL signs include **repetition as part of their form**.\n"
        "  * Examples: '#DO', '#NO', '#HA'—these signs repeat movement to differentiate them from standard fingerspelling.\n\n"

        " ** Modulation: with Repetition for Enhanced Emphasis\n"
        "  * When repeating a sign for emphasis, signers can **modify signing speed, facial expressions, or movement size**.\n"
        "  * Larger or more exaggerated repetitions may **intensify emotion**.\n"
        "Do not reuse the examples below. They are just for reference. \n\n"
        " **Examples:**\n"
        "  - English: 'It was so amazing!'\n"
        "  - ASL: 'WOW++ LOOK-AT WONDER+++ (prosodic emphasis).'\n\n"

        "***How to Apply Repetition in ASL Translation:*** \n"
        " * If a concept in English uses words like 'very', 'a lot', 'all the time', or 'constantly,' consider using **lexical or morphological repetition**.\n"
        " * If a word represents **strong emotion** (e.g., 'cry,' 'love,' 'excited'), consider using **reduplication for emphasis**.\n"
        " * If an action is **habitual or repeated**, adjust the signs **movement pattern to reflect iteration**.\n\n"

        "Do not reuse the examples below. They are just for reference. \n\n"

        "### Examples ###\n"
        # add 5 more repetition-based examples here

        "\n\n"

        "### Step-by-Step Translation Guide ###\n"
        "1. Identify and front any time markers.\n"
        "2. Identify the sentence topic and place it before the comment.\n"
        "3. If it's a WH-question, move the WH-word to the end.\n"
        "4. Remove auxiliary verbs, articles, and unnecessary prepositions.\n"
        "5. Use repetition for emphasis, if needed.\n\n"

        "Important: Do not include any example text in your answer.\n\n"
        "Now, translate the following sentence into a fluent ASL gloss using the rules above:\n\n"
        "Keep the response short.\n\n"

        f"English: {english_response}\n"
        "ASL:"
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert ASL interpreter. Translate the following English sentence into ASL gloss using correct ASL grammar and structure."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100  # Increase this if ASL output is too short
        )

        asl_like_response = response['choices'][0]['message']['content'].strip()

        # Optionally simplify the result if needed
        # simplified_response = simplify_to_asl_grammar(asl_like_response)
        final_gloss = postprocess_asl_gloss(asl_like_response, wlasl_words)

        return final_gloss

    except Exception as e:
        return f"[OpenAI API Error: {e}]"

def main():
    # File path for input
    input_file_path = "chatbot/ASL_to_Text.txt"
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

# # Testing the full pipeline
# if __name__ == "__main__":
#     print("\n Running test for full ASL to Gloss pipeline:\n")

#     # Test Cases
#     test_cases = [
#         "how fix computer broken",
#         "dog bark loud night",
#         "where find bus stop near",
#         "friend visit home weekend",
#         "mother cook dinner family",
#     ]

#     # Run each test case through the full pipeline
#     for test_input in test_cases:
#         print(f"\nASL Input: {test_input}")

#         # Step 1: Interpret ASL structure into English
#         english_interpretation = interpret_asl_input(test_input)
#         print(f"English Interpretation: {english_interpretation}")

#         # Step 2: Generate chatbot response in English
#         english_response = generate_english_response(english_interpretation)
#         print(f"English Response: {english_response}")

#         # Step 3: Convert chatbot response into ASL gloss
#         final_gloss = generate_asl_response(english_response)
#         print(f"Final ASL Gloss: {final_gloss}")

if __name__ == "__main__":
    main()
