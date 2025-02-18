import re

def process_text(input_text):
    # Split the input text into lines
    lines = input_text.split("\n")
    
    # Iterate over each line and modify it
    processed_lines = []
    for line in lines:
        # Remove timestamp pattern (like 00:09:55) if present
        line = re.sub(r'^\d{2}:\d{2}:\d{2}', '', line).strip()
        # Add "type in" after each new line
        if line:  # Check if the line is not empty after removal
            processed_lines.append(line + " type in")
    
    # Join the processed lines back into a single string
    return "\n".join(processed_lines)

# Example input text
# input_text = """
# 00:09:55	we can say torch dot zeros so this will put all um zeros in it or we can say torch dot once so this will put once in all the um items um then we can also give it a specific data type so first of all we can have a look at the data type by saying x dot d type so if we run this then we see by default it's a float 32 but we can also give it the d type parameter and here we can say for example torch dot in so now it's all integers or we can say torch dot double now it is doubles um or we can also say

# 00:10:51	for example float 16 just um yeah and now if you want to have a look at the size we can do this by saying x dot size and this is a function so we have to use parentheses so this will print the size of it and we can also construct a tensor from data so for example from a python list so for example here we can say x equals torch dot 10 sore and then here we put a list with some elements so let's say 2.5 0.1 and then print our tensor so this is also how we can create a tensor and now let's talk about some basic
# """
with open('my_ai_model/data/conversations.txt', 'r') as file:
    # Read the contents of the file
    input_text = file.read()
    
# Process the text
output_text = process_text(input_text)

with open('my_ai_model/data/clean.txt', 'a') as file:
    file.write(output_text)

print("done 👍")
