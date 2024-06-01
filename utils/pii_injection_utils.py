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
from .pii_prompt_utils import (
    get_data_transition_prompt,
    get_text_transition_prompt,
    process_generated_text
)

nltk.download('punkt')

def generate_data_transition(model, tokenizer, sampling_params, text, data, data_type):
    prompt = get_data_transition_prompt(tokenizer, text, data, data_type)

    preds = model.generate(prompts=[prompt], sampling_params=sampling_params)
    output = process_generated_text(preds[0].outputs[0].text)
    return output


def generate_text_transition(model, tokenizer, sampling_params, first_text, second_text):
    prompt = get_text_transition_prompt(tokenizer, first_text, second_text)

    preds = model.generate(prompts=[prompt], sampling_params=sampling_params)
    output = process_generated_text(preds[0].outputs[0].text)
    return output

def generate_transitions(model, sampling_params, prompts):
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
        transition_before = output
        first_text = f"{first_text} {transition_before} {pii}"

        output = generate_text_transition(first_text=first_text, second_text=second_text)
        transition_after = output

        splitted_text.insert(pii_insert_index, f"{transition_before} {pii} {transition_after}")
    
    pii_text = ' '.join(splitted_text)

    return pii_text