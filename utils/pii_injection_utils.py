import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import random
import re
from tqdm import tqdm
from wonderwords import RandomWord
import json
import sys
import time
import nltk

nltk.download('punkt')

default_max_characters = 4000

generate_data_transition_prompt = '''You are given a text, data, and the type of the data. Your task is to connect the given text with the provided data by writing a transition between them. The transition should expand upon the given text and coherently connect the text to the data. If the text and data can be coherently connected without a transition, set the value of the key "transition" equal to an empty string like this "". Be sure to add punctuation where necessary.
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

fill_transition_prompt = '''You are given two text. Your task is to connect those two text by writing a transition between them. The transition should expand upon the given first text and coherently connect the first text to the the second text. If the first and second texts can be coherently connected without a transition, set the value of the key "transition" equal to an empty string like this "". Be sure to add punctuation where necessary.
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

def get_data_transition_prompt(tokenizer, text, data, data_type, max_characters=default_max_characters):
    if len(text) > max_characters:
        text = text[-max_characters:]
    messages = [
    {"role": "system", "content": "You a helpful and honest assistant."},
    {"role": "user", "content": generate_data_transition_prompt.format(text=text, data=data, data_type=data_type)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True,
    )

    return prompt

def get_text_transition_prompt(tokenizer, first_text, second_text, max_characters=default_max_characters):
    if len(first_text) > max_characters:
        first_text = first_text[-max_characters:]

    if len(second_text) > max_characters:
        second_text = second_text[:-max_characters]
    messages = [
        {"role": "system", "content": "You a helpful and honest assistant."},
        {"role": "user", "content": fill_transition_prompt.format(first_text=first_text, second_text=second_text)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True,
    )

    return prompt

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
        
    return output
    

def generate_data_transition(model, sampling_params, text, data, data_type):
    prompt = get_data_transition_prompt(text, data, data_type)

    preds = model.generate(prompts=[prompt], sampling_params=sampling_params)
    output = process_generated_text(preds[0].outputs[0].text)
    return output


def generate_text_transition(model, sampling_params, first_text, second_text):
    prompt = get_text_transition_prompt(first_text, second_text)

    preds = model.generate(prompts=[prompt], sampling_params=sampling_params)
    output = process_generated_text(preds[0].outputs[0].text)
    return output

def generate_outputs(model, sampling_params, prompts):
    preds = model.generate(prompts=prompts, sampling_params=sampling_params)
    outputs = [process_generated_text(pred.outputs[0].text) for pred in preds]

    return outputs


def get_pii_list(pii_map):
    pii_list = []

    for label in pii_map:
        if not pii_map[label]: continue
        pii_list.extend([(value, label) for value in pii_map[label]])

    # Randomly change order of pii_list
    random.shuffle(pii_list)

    return pii_list

def generate_pii_text(text, pii_map):

    # Divide text into sentences with nltk
    sentences = nltk.sent_tokenize(text)

    # Randomly choose sentences to use as tex
    num_sentences = random.randint(1, len(sentences))
    sentences = random.sample(sentences, num_sentences)

    splitted_text = text.split(' ')
    pii_list = get_pii_list(pii_map)

    for (pii, label) in pii_list:
        pii_insert_index = random.randint(0, len(splitted_text))
        first_text = ' '.join(splitted_text[:pii_insert_index])
        second_text = ' '.join(splitted_text[pii_insert_index:])

        output = generate_data_transition(text=first_text, data=pii, data_type=label)
        transition_before = output['transition']
        first_text = f"{first_text} {transition_before} {pii}"

        output = generate_text_transition(first_text=first_text, second_text=second_text)
        transition_after = output['transition']

        splitted_text.insert(pii_insert_index, f"{transition_before} {pii} {transition_after}")
    
    pii_text = ' '.join(splitted_text)

    return pii_text