def convert_text_file_format(input_file, output_file):
    """
    Converts the examples in a text file from a standard format to a Python string format
    that can be directly copied and pasted into prompting functions, ensuring correct newline spacing
    with an extra blank line between examples.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    formatted_lines = []
    example_num = 51
    temp_buffer = []

    for line in lines:
        line = line.strip()

        if line.startswith("Example"):
            if temp_buffer:
                formatted_lines.append("\n".join(temp_buffer))  # Append previous example without extra newlines
                temp_buffer = []

            temp_buffer.append(f'        "**Example {example_num}:**\\n"')
            example_num += 1

        elif line.startswith("English:"):
            english_text = line.replace('English:', '').strip().strip('"')
            temp_buffer.append(f'        "English: \'{english_text}\'\\n"')

        elif line.startswith("ASL:"):
            asl_text = line.replace('ASL:', '').strip().strip('"')
            temp_buffer.append(f'        "ASL: \'{asl_text}\'\\n\\n"')  # Correctly formatted newline

    if temp_buffer:
        formatted_lines.append("\n".join(temp_buffer))  # Ensure last example is spaced properly

    # Join all examples with exactly one blank line between them
    formatted_text = "\n\n".join(formatted_lines) + "\n"

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(formatted_text)

    print(f"Conversion complete! Output saved to {output_file}")

# Example Usage:
input_text_file = "sets_og/set2.txt"  # Replace with the actual input file path
output_text_file = "outputs/set2_output.txt"

# Convert the file format
convert_text_file_format(input_text_file, output_text_file)
