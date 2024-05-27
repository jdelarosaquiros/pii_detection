import os
import argparse
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
from datasets import load_dataset
from utils.pii_generation_utils import (
    pii_labels,
    generate_name_samples,
    generate_email_samples,
    generate_username_samples,
    generate_id_number_samples,
    generate_phone_number_samples,
    generate_street_address_samples,
    generate_url_samples,
)
from utils.pii_injection_utils import (
    get_pii_list,
    generate_outputs,
    get_data_transition_prompt,
    get_text_transition_prompt,
)

def generate_pii_samples(raw_data_path: str):
    # Load the raw data
    pii_raw_data = pd.read_csv(raw_data_path)
    pii_raw_data.columns = pii_raw_data.columns.str.strip() # Delete leading space from column names in pii_raw_data dataframe

    # Iterate through each row in the pii_raw_data dataframe and generate all samples for each row
    name_samples = []
    email_samples = []
    username_samples = []
    id_number_samples = []
    phone_number_samples = []
    street_address_samples = []
    url_samples = []

    for first, mi, last, name, phone, email, guid, digit, street, state, zip, zip9, domain, paragraph, sentence, word, alpha in tqdm(pii_raw_data.values):
        name_samples.extend(generate_name_samples(first, mi, last, name))
        email_samples.extend(generate_email_samples(first, last, email))
        username_samples.extend(generate_username_samples(name, email))
        id_number_samples.append(generate_id_number_samples(digit))
        phone_number_samples.append(generate_phone_number_samples(phone))
        street_address_samples.extend(generate_street_address_samples(street, state, zip, zip9))
        url_samples.append(generate_url_samples(domain))

    # Save all samples to a json file
    pii_samples = {
        "name": name_samples,
        "email": email_samples,
        "username": username_samples,
        "id_number": id_number_samples,
        "phone_number": phone_number_samples,
        "street_address": street_address_samples,
        "url": url_samples
    }

    with open('pii_samples.json', 'w') as f:
        json.dump(pii_samples, f)

    return pii_samples

def group_pii_samples(pii_samples: dict, max_samples_per_type: int = 5):
    # Get the samples for each PII label
    name_samples: list[str] = pii_samples['name']
    email_samples: list[str] = pii_samples['email']
    username_samples: list[str] = pii_samples['username']
    id_number_samples: list[str] = pii_samples['id_number']
    phone_number_samples: list[str] = pii_samples['phone_number']
    street_address_samples: list[str] = pii_samples['street_address']
    url_samples: list[str] = pii_samples['url']

    # Initialize samples remaining
    pii_labels_remaining = pii_labels.copy()
    name_samples_remaining = name_samples.copy()
    email_samples_remaining = email_samples.copy()
    username_samples_remaining = username_samples.copy()
    id_number_samples_remaining = id_number_samples.copy()
    phone_number_samples_remaining = phone_number_samples.copy()
    street_address_samples_remaining = street_address_samples.copy()
    url_samples_remaining = url_samples.copy()


    # Loops until there is no samples left and does the following:
    # 1. Randomly decide which sample type to use
    # 2. Choose a random number of samples from each selected sample type
    # 3. Remove the selected samples from the list of samples.

    grouped_pii_samples = []
    total_samples = len(name_samples_remaining) + len(email_samples_remaining) + len(username_samples_remaining) + len(id_number_samples_remaining) + len(phone_number_samples_remaining) + len(street_address_samples_remaining) + len(url_samples_remaining)

    with tqdm(total=total_samples) as pbar:
        while name_samples_remaining or email_samples_remaining or username_samples_remaining or id_number_samples_remaining or phone_number_samples_remaining or street_address_samples_remaining or url_samples_remaining:
            selected_name_samples = []
            selected_email_samples = []
            selected_username_samples = []
            selected_id_number_samples = []
            selected_phone_number_samples = []
            selected_street_address_samples = []
            selected_url_samples = []
            
            selected_labels = random.choices(pii_labels_remaining, k = random.randint(1, len(pii_labels_remaining)))

            if 'STUDENT_NAME' in selected_labels and name_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(name_samples_remaining)))
                selected_name_samples = random.sample(name_samples_remaining, num_samples)
                name_samples_remaining = [sample for sample in name_samples_remaining if sample not in selected_name_samples]
            
            if 'EMAIL' in selected_labels and email_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(email_samples_remaining)))
                selected_email_samples = random.sample(email_samples_remaining, num_samples)
                email_samples_remaining = [sample for sample in email_samples_remaining if sample not in selected_email_samples]
            
            if 'USERNAME' in selected_labels and username_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(username_samples_remaining)))
                selected_username_samples = random.sample(username_samples_remaining, num_samples)
                username_samples_remaining = [sample for sample in username_samples_remaining if sample not in selected_username_samples]
            
            if 'ID_NUMBER' in selected_labels and id_number_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(id_number_samples_remaining)))
                selected_id_number_samples = random.sample(id_number_samples_remaining, num_samples)
                id_number_samples_remaining = [sample for sample in id_number_samples_remaining if sample not in selected_id_number_samples]
            
            if 'PHONE_NUMBER' in selected_labels and phone_number_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(phone_number_samples_remaining)))
                selected_phone_number_samples = random.sample(phone_number_samples_remaining, num_samples)
                phone_number_samples_remaining = [sample for sample in phone_number_samples_remaining if sample not in selected_phone_number_samples]
            
            if 'STREET_ADDRESS' in selected_labels and street_address_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(street_address_samples_remaining)))
                selected_street_address_samples = random.sample(street_address_samples_remaining, num_samples)
                street_address_samples_remaining = [sample for sample in street_address_samples_remaining if sample not in selected_street_address_samples]
            
            if 'PERSONAL_URL' in selected_labels and url_samples_remaining:
                num_samples = random.randint(1, min(max_samples_per_type, len(url_samples_remaining)))
                selected_url_samples = random.sample(url_samples_remaining, num_samples)
                url_samples_remaining = [sample for sample in url_samples_remaining if sample not in selected_url_samples]

            # Add selected samples to sample_outputs
            grouped_pii_samples.append({'name': selected_name_samples, 'email': selected_email_samples, 'username': selected_username_samples, 'id_number': selected_id_number_samples, 'phone_number': selected_phone_number_samples, 'street_address': selected_street_address_samples, 'url': selected_url_samples})

            # Remove pii labels with no remaining samples
            if not name_samples_remaining and 'STUDENT_NAME' in pii_labels_remaining:
                pii_labels_remaining.remove('STUDENT_NAME')
            if not email_samples_remaining and 'EMAIL' in pii_labels_remaining:
                pii_labels_remaining.remove('EMAIL')
            if not username_samples_remaining and 'USERNAME' in pii_labels_remaining:
                pii_labels_remaining.remove('USERNAME')
            if not id_number_samples_remaining and 'ID_NUMBER' in pii_labels_remaining:
                pii_labels_remaining.remove('ID_NUMBER')
            if not phone_number_samples_remaining and 'PHONE_NUMBER' in pii_labels_remaining:
                pii_labels_remaining.remove('PHONE_NUMBER')
            if not street_address_samples_remaining and 'STREET_ADDRESS' in pii_labels_remaining:
                pii_labels_remaining.remove('STREET_ADDRESS')
            if not url_samples_remaining and 'PERSONAL_URL' in pii_labels_remaining:
                pii_labels_remaining.remove('PERSONAL_URL')
            
            # Update progress bar
            num_samples_selected = len(selected_name_samples) + len(selected_email_samples) + len(selected_username_samples) + len(selected_id_number_samples) + len(selected_phone_number_samples) + len(selected_street_address_samples) + len(selected_url_samples)
            pbar.update(num_samples_selected)

    # Save the sample outputs to a json file
    with open('grouped_pii_samples.json', 'w') as f:
        json.dump(grouped_pii_samples, f)

    return grouped_pii_samples

def generate_pii_dataset(model, tokenizer, sampling_params, texts: list[str], grouped_pii_samples: list[dict], output_dataset_name: str, max_dataset_size: int = None):
    pii_dataset = []

    pii_dataset_length = min(len(texts), len(grouped_pii_samples))
    if max_dataset_size:
        pii_dataset_length = min(pii_dataset_length, max_dataset_size)

    all_pii_lists = [get_pii_list(pii_map) for pii_map in grouped_pii_samples[:pii_dataset_length]]

    # Randomly choose the size of each text
    for i, text in enumerate(texts):
        # Divide text into sentences with nltk
        sentences = nltk.sent_tokenize(text)

        # Randomly choose sentences to use as tex
        num_sentences = random.randint(1, len(sentences))
        sentences = random.sample(sentences, num_sentences)

        texts[i] = ' '.join(sentences)


    # Randomly split text by sentence or by word
    all_split_by_sentence = [random.choice([True, False]) for _ in range(len(texts))]
    all_splitted_texts = [nltk.sent_tokenize(text) if split_by_sentence else text.split(' ') for text, split_by_sentence in zip(texts, all_split_by_sentence)]

    # Iterate over all indices of the pii lists
    max_pii_list_length = max([len(pii_list) for pii_list in all_pii_lists])

    for index in range(max_pii_list_length):
        remaining_pii_lists = [pii_list[index] for pii_list in all_pii_lists if index < len(pii_list)]
        remaining_splitted_texts = [splitted_text for splitted_text, pii_list in zip(all_splitted_texts, all_pii_lists) if index < len(pii_list)]

        pii_insertion_indexes = [random.randint(0, len(splitted_text)) for splitted_text in remaining_splitted_texts]

        first_texts = [' '.join(splitted_text[:pii_insert_index]) for splitted_text, pii_insert_index in zip(remaining_splitted_texts, pii_insertion_indexes)]
        second_texts = [' '.join(splitted_text[pii_insert_index:]) for splitted_text, pii_insert_index in zip(remaining_splitted_texts, pii_insertion_indexes)]

        data_transition_prompts = [get_data_transition_prompt(tokenizer, first_text, pii, label) for first_text, (pii, label) in zip(first_texts, remaining_pii_lists)]
        outputs = generate_outputs(model, sampling_params, data_transition_prompts)
        
        transitions_before = [output['transition'] for output in outputs]

        first_texts = [f"{first_text} {transition} {pii}" for first_text, transition, (pii, _) in zip(first_texts, transitions_before, remaining_pii_lists)]

        text_transition_prompts = [get_text_transition_prompt(tokenizer, first_text, second_text) for first_text, second_text in zip(first_texts, second_texts)]
        outputs = generate_outputs(model, sampling_params, text_transition_prompts)

        transitions_after = [output['transition'] for output in outputs]


        print(f"Inserting PII: {index+1}/{max_pii_list_length}")
        for splitted_text, pii_insert_index, transition_before, (pii, label), transition_after, split_by_sentence in zip(remaining_splitted_texts, pii_insertion_indexes, transitions_before, remaining_pii_lists, transitions_after, all_split_by_sentence):
            if split_by_sentence and transition_after[0].isupper():
                first_text_sep_char = '. '
            else:
                first_text_sep_char = ' '

            second_text_sep_char = '.' if split_by_sentence else ''
            splitted_text.insert(pii_insert_index, f"{transition_before} {pii}{first_text_sep_char}{transition_after}{second_text_sep_char}")

    # Generated text with pii
    pii_texts = [' '.join(splitted_text) for splitted_text in all_splitted_texts]


    # Create json of all data
    for source_text, pii_text, pii_list in zip(texts, pii_texts, all_pii_lists):
        pii_data = [values for values, _ in pii_list]
        pii_labels = [label for _, label in pii_list]
        pii_dataset.append({'source_text': source_text, 'pii_text': pii_text, 'pii_data': pii_data, 'pii_labels': pii_labels})

    # Save the pii dataset to a json file
    with open(output_dataset_name, 'w') as f:
        json.dump(pii_dataset, f)

    return pii_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dataset_name_path', type=str, default="pii_dataset")
    parser.add_argument('--pii_raw_data', type=str, default="pii_raw_data.csv")
    parser.add_argument('--grouped_pii_samples_path', type=str, default=None)
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--gpu_util', type=float, default=0.9)
    parser.add_argument("--world_size",  type=int, default=1, help="world size to use multiple GPUs.")
    args = parser.parse_args()

    # Generate or Load PII samples
    if not args.grouped_pii_samples_path:
        pii_samples = generate_pii_samples(args.pii_raw_data)
        grouped_pii_samples = group_pii_samples(pii_samples)
    else:
        with open(args.grouped_pii_samples_path, 'r') as f:
            grouped_pii_samples = json.load(f)

    # Load the model and tokenizer
    model = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, tensor_parallel_size=args.world_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    terminators = [tokenizer.eos_token]

    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.9, 
        max_tokens=2058, 
        skip_special_tokens=True,
        stop=terminators
    )

    # Load texts
    essay_dataset = load_dataset("qwedsacf/ivypanda-essays")

    # Generate PII Dataset
    pii_dataset = generate_pii_dataset(model, tokenizer, sampling_params, essay_dataset['train']['TEXT'], grouped_pii_samples, args.output_dataset_name_path, max_dataset_size = args.max_data_size)

    print(f"\nFinished Generating PII Dataset with {len(pii_dataset)} samples\n")

    # Print the first 5 samples
    print("First 5 samples:")
    for i in range(5):
        if i >= len(pii_dataset):
            break
        print(f"Sample {i+1}:")
        print(f"\nSource Text: {pii_dataset[i]['source_text']}")
        print(f"\nPII Text: {pii_dataset[i]['pii_text']}")
        print(f"\nPII Data: {pii_dataset[i]['pii_data']}")
        print(f"\nPII Labels: {pii_dataset[i]['pii_labels']}")
        print("\n")


if __name__ == "__main__":
    main()