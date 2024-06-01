import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
import json
import nltk
import argparse
from termcolor import colored
from datasets import load_dataset
from huggingface_hub import login
from generate_pii_dataset import generate_pii_dataset
from utils.pii_injection_utils import (
    get_pii_list,
    generate_data_transition,
    generate_text_transition,
)

# This function inserts PII into the text
# Modify this function to change how the PII is inserted into the text
def generate_pii_text(model, tokenizer, sampling_params, text, pii_list, split_by_sentence=True, max_piis=None):
    splitted_text = nltk.sent_tokenize(text) if split_by_sentence else text.split(' ')

    for i, (pii, label) in enumerate(pii_list):
        if max_piis is not None and i >= max_piis:
            break

        pii_insert_index = random.randint(0, len(splitted_text))
        first_text = ' '.join(splitted_text[:pii_insert_index])
        second_text = ' '.join(splitted_text[pii_insert_index:])

        transition = generate_data_transition(model, tokenizer, sampling_params, first_text, data=pii, data_type=label)
        transition_before = transition
        first_text = f"{first_text} {colored(transition_before, 'green')} {pii}"

        transition = generate_text_transition(model, tokenizer, sampling_params, first_text=first_text, second_text=second_text)
        transition_after = transition

        splitted_text.insert(pii_insert_index, f"{colored(transition_before, 'green')} {colored(pii, 'blue')} {colored(transition_after, 'green')}")
    
    pii_text = ' '.join(splitted_text)

    return pii_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouped_pii_samples_path', type=str, default="grouped_pii_samples.json")
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") # Note: If using Llama, you should export your huggingface token: export HF_TOKEN=<your token>
    parser.add_argument('--gpu_util', type=float, default=0.9)
    parser.add_argument("--world_size",  type=int, default=1, help="world size to use multiple GPUs.")
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--text_index', type=int, default=0)
    parser.add_argument('--max_piis', type=int, default=None)

    args = parser.parse_args()

    # Select which sample and text to test
    sample_index = args.sample_index
    text_index = args.text_index
    max_piis = args.max_piis # Set to None to inject all PIIs or set to a number to inject a maximum number of PIIs

    # Load the model and tokenizer
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 

    model = LLM(model=model_id, gpu_memory_utilization=0.9, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate or load PII samples
    with open(args.grouped_pii_samples_path, 'r') as f:
        grouped_pii_samples = json.load(f)

    # Load source texts
    essay_dataset = load_dataset("qwedsacf/ivypanda-essays")

    # Set pii sample
    pii_sample = grouped_pii_samples[sample_index]

    # Set text to be injected with PII
    text = essay_dataset['train'][text_index]['TEXT']
    print(f"{'#'*15} Text {'#'*15}")
    print(text)
    print("#" * max([len(line) for line in text.split('\n')]))

    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.9, 
        max_tokens=2058, 
        skip_special_tokens=True,
        stop=[tokenizer.eos_token]
    )

    pii_list = get_pii_list(pii_sample)
    print(f"{'#'*15} PII List {'#'*15}")
    print(pii_list)
    print("#" * max([len(line) for line in str(pii_list).split('\n')]))

    pii_text = generate_pii_text(model, tokenizer, sampling_params, text, pii_list, split_by_sentence=False, max_piis=max_piis)
    print(f"{'#'*15} PII Text {'#'*15}")
    print(pii_text)
    print("#" * max([len(line) for line in pii_text.split('\n')]))

if __name__ == "__main__":
    main()