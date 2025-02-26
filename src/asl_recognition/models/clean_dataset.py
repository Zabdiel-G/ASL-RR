import pandas as pd
import re
import csv
# Load valid words from the text file
with open("wlasl_class_list.txt", "r", encoding="utf-8") as file:
    sample = file.read(500)  # Read the first 500 characters
    dialect = csv.Sniffer().sniff(sample)  # Detect delimiter
    detected_delimiter = dialect.delimiter
    print("Detected delimiter:", repr(detected_delimiter))

# Step 2: Read the second column and store valid words
valid_words = set()

with open("wlasl_class_list.txt", "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=detected_delimiter)
    for row in reader:
        if len(row) > 1:  # Ensure the second column exists
            valid_words.add(row[1])  # Store only the second column values

# Step 3: Print the valid words
print("\nValid words from second column:")
print(list(valid_words)[:20])

# Function to clean the sentence:
# - Remove words containing "X-"
# - Remove "DESC-"
# - Remove all instances of ".", ",", and '"'
def clean_sentence(sentence):
    sentence = str(sentence).strip() 

    words = sentence.split()
    words = [word for word in words if "-" not in word]  
    sentence = " ".join(words)


    sentence = sentence.replace("DESC-", "")

    sentence = re.sub(r"\bwh-q\((.*?)\)", r"\1", sentence) 
    sentence = re.sub(r"\([^)]*\)", "", sentence) 

    sentence = sentence.replace(".", "").replace(",", "").replace('"', "")

    return sentence.strip().lower() if sentence else None

# Function to check if a sentence contains only valid words (after cleanup)
def is_valid_sentence(sentence):
    words = str(sentence).split()  # Split sentence again after cleanup
    return all(word in valid_words for word in words)  # Keep only if all words are valid

df1 = pd.read_parquet("https://huggingface.co/datasets/achrafothman/aslg_pc12/resolve/main/data/train-00000-of-00001.parquet")
df2 = pd.read_csv("corpus_0001.clean.asl.txt", delimiter="\t", header=None, names=["gloss"]) 

print("df1 columns:", df1.columns)
print("df2 columns:", df2.columns)
print(df1.head())  # See a preview of df1
print(df2.head())  # See a preview of df2

df = pd.concat([df1, df2], ignore_index=True)

gloss_column = "gloss"  

df = df[[gloss_column]].copy()

df[gloss_column] = df[gloss_column].apply(clean_sentence)

df.to_csv("filtered_dataset_cleaned.csv", index=False, header=False)

filtered_df = df[df[gloss_column].apply(is_valid_sentence)]

filtered_df.to_csv("filtered_dataset1.csv", index=False)

filtered_df[gloss_column].to_csv("filtered_dataset1.txt", index=False, header=False, sep="\n")

print("Filtered Dataset:")
print(filtered_df)

print("\nCleaned Dataset:")
print(df)


