import re
import json

default_max_characters = 4000 # Equivalent to around 2000 tokens

# Prompt for generating a transition between a given source text, data, and type of data.
# Note: the text could end with a complete or partial sentence, and the transition should connect it to the data.
data_transition_prompt = '''You are given a text, data, and the type of the data. Your task is to connect the given text with the provided data by writing a transition between them. The transition should expand upon the given text and coherently connect the text to the data. If the text and data can be coherently connected without a transition, set the value of the key "transition" equal to an empty string like this "". Be sure to add punctuation where necessary.
Your response MUST be a dictionary in JSON. The dictionary should always contain the following keys: "text" as the given text, "transition" as the transition between text and data, the type of the data as "data type", and the data as "data".
You MUST only respond in the format as described below. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED. START YOUR RESPONSE WITH '{{'. END YOUR RESPONSE WITH '}}'.
[response format]: {{"text": "[text]", "transition": "[transition]", "data type": "[data type]", "data": "[data]"}}

Here is one example:
[text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will
[data type]: name
[data]: Juan Manuel
[response]: {{"text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will", "transition": "face the one named" "data type": "name", "data": "Juan Manuel" }}

Now complete the following, RESPONSE SHOULD ONLY BE IN JSON MAP FORMAT, NO OTHER WORDS!!!:
[text]: {text}
[data]: {data}
[data type]: {data_type}
[response]: '''

# Prompt for generating a transition between two texts. The first text is the output from data_transition_prompt (source text + transition + data), and it should be connected to the second text with a transition.
# Note: the first and second text could end with a complete or partial sentence, and the transition should connect them regardless.
text_transition_prompt = '''You are given two text. Your task is to connect those two text by writing a transition between them. The transition should expand upon the given first text and coherently connect the first text to the the second text. If the first and second texts can be coherently connected without a transition, set the value of the key "transition" equal to an empty string like this "". Be sure to add punctuation where necessary.
Your response MUST be a dictionary in JSON. The dictionary should contain the following keys: "first text" as the first text, "transition" as the transition between first and second texts, and "second text" as the second text.
You MUST only respond in the format as described below. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED. START YOUR RESPONSE WITH '{{'. END YOUR RESPONSE WITH '}}'.
[response format]: {{"first text": "[first text]", "transition": "[transition]", "second text": "[second text]"}}

Here is one example:
[first text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will face Juan Manuel
[second text]: either Rafael Nadal or Novak Djokovic in the final.
[response]: {{"first text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will face Juan Manuel", "transition": "who is looking to win his first ATP title, in the final match against", "second text": "either Rafael Nadal or Novak Djokovic in the final." }}

Now complete the following, RESPONSE SHOULD ONLY BE IN JSON MAP FORMAT, NO OTHER WORDS!!!:
[first text]: {first_text}
[second text]: {second_text}
[response]: '''

# List of input and output examoples for data_transition_prompt
data_few_shot_examples = [
{
    "text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will",
    "data": "Juan Manuel",
    "data_type": "name",
    "transition": "face the one named",
    "response": '''{{"text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will", "transition": "face the one named" "data type": "name", "data": "Juan Manuel" }}'''

},
]

# List of input and output examoples for text_transition_prompt
text_few_shot_examples = [
{
    "first_text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will face Juan Manuel",
    "second_text": "either Rafael Nadal or Novak Djokovic in the final.",
    "transition": "who is looking to win his first ATP title, in the final match against",
    "response": '''{{"first text": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will face Juan Manuel", "transition": "who is looking to win his first ATP title, in the final match against", "second text": "either Rafael Nadal or Novak Djokovic in the final." }}'''

},
]

# Function to generate a prompt for appending a data to a text by writing a transition between them.
def get_data_transition_prompt(tokenizer, text, data, data_type, max_characters=default_max_characters):
    if len(text) > max_characters:
        text = text[-max_characters:]

    messages = [{"role": "system", "content": "You a helpful and honest assistant."},]

    ## Uncomment the following lines to add few-shot examples to the prompt
    ## Note: This is the correct way to add few-shot examples to the prompt, but I added them directly in the prompt becuase it lead to better results for some reason. If to do it this way, you need to remove the examples from the prompt.
    # for example in data_few_shot_examples:
    #     messages.append({"role": "user", "content": data_transition_prompt.format(text=example["text"], data=example["data"], data_type=example["data_type"])})
    #     messages.append({"role": "assistant", "content": example["response"]})

    messages.append({"role": "user", "content": data_transition_prompt.format(text=text, data=data, data_type=data_type)})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True,
    )

    return prompt

# Function to generate a prompt for appending a text to another text by writing a transition between them.
def get_text_transition_prompt(tokenizer, first_text, second_text, max_characters=default_max_characters):
    if len(first_text) > max_characters:
        first_text = first_text[-max_characters:]

    if len(second_text) > max_characters:
        second_text = second_text[:-max_characters]

    messages = [{"role": "system", "content": "You a helpful and honest assistant."}]

    ## Uncomment the following lines to add few-shot examples to the prompt
    ## Note: This is the correct way to add few-shot examples to the prompt, but I added them directly in the prompt becuase it lead to better results for some reason. If to do it this way, you need to remove the examples from the prompt.
    # for example in text_few_shot_examples:
    #     messages.append({"role": "user", "content": text_transition_prompt.format(first_text=example["first_text"], second_text=example["second_text"])})
    #     messages.append({"role": "assistant", "content": example["response"]})
    
    messages.append({"role": "user", "content": text_transition_prompt.format(first_text=first_text, second_text=second_text)})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True,
    )

    return prompt

# Function to process the generated text from the model, and it should return the transition
# Note: Make sure this function can handle the outputs from the both generate_data_transition_prompt and generate_text_transition_prompt by structuring their output in the same way. In this case, They both return a JSON object with the key "transition"
def process_generated_text(generated_text):
    generated_text = generated_text.replace('\n', '').strip()
    default_output = {"transition": ""}

    if not generated_text:
        return default_output

    try:
        output = json.loads(generated_text)
    except:
        if('{' in generated_text and '}' not in generated_text):
            output = re.sub(r'"?$', '"}', generated_text)
            output = re.sub(r'("})+', '"}', output)
            try:
                output = json.loads(output)
            except:
                print(f'Unexpected Error: {generated_text}')
                output = default_output
        else:
            output = default_output
    

    if 'transition' not in output:
        output = default_output
        
    return output['transition']