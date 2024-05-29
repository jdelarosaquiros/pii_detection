from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import random
import spacy
from wonderwords import RandomWord
from SID_Dataset_Generator.utils.utils import json_pii_formatter, extract_json_like_objects
from SID_Dataset_Generator.templates.templates import StandardAnswer_StudentPII

class Dataset:
    def __init__(self, llm=None, llm_tokenizer=None, nlp_tokenizer=None, assembler=None, random_data_gen=None, formatter = None):
        if llm is not None:
            self.model_name = llm
            self.model = self._initial_model(self.model_name)
        else:
            self.model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
            self.model = self._initial_model(self.model_name)

        if llm_tokenizer is not None:
            self.tokenizer = llm_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if nlp_tokenizer is not None:
            self.nlp_tokenizer = nlp_tokenizer
        else:
            self.nlp_tokenizer = spacy.load("en_core_web_sm")

        if random_data_gen is not None:
            self.random_data_gen = random_data_gen
        else:
            self.random_data_gen = RandomWord()

        if assembler is not None:
            self.Assem = assembler(llm_model=self.model, tokenizer=self.tokenizer, formatter=formatter)
        else:
            self.Assem = None

    def _initial_model(self, model_name):
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return None

        model = LLM(
            model_name,
            gpu_memory_utilization=0.90,
            tensor_parallel_size=gpu_count,
        )

        return model

    def generate(self, documents:list, piis:list, answer_formatter = None, category='document'):
        if answer_formatter is None:
            answer_formatter = StandardAnswer_StudentPII()
            answer_formatter_func = answer_formatter.format_prompt
        else:
            answer_formatter = answer_formatter()
            answer_formatter_func = answer_formatter.format_prompt

        return self.Assem.document_assembler(documents, piis, answer_formatter_func, category)

class Assembler:
    def __init__(self, llm_model, tokenizer, formatter):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.role = 'assembler'

    def document_assembler(self, document_list:list, piis_list:list, answer_formatter, category='document'):
        data = []
        assert len(document_list) == len(piis_list)

        for i in range(len(document_list)):
            title = self._generate_title(document_list[i])
            piis = self._format_pii(piis_list[i])
            cot_piis = self._generate_aggregated_info(piis, answer_formatter)

            if cot_piis == []:
                continue

            header = self._generate_header(piis)
            full_essay = self._create_full_essay(title, header, document_list[i])
            full_instruction = self.formatter.format_prompt(full_essay, self.tokenizer.eos_token, cot_piis)
            half_instruction = self.formatter.format_prompt(full_essay, self.tokenizer.eos_token, '')

            data.append({
                'category': category,
                'content': full_essay,
                'training_text': full_instruction,
                'testing_text': half_instruction,
                'answer': cot_piis
            })

        return data

    def _generate_title(self, document):
        print(type(document))
        print(document)
        messages = [
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": 'Create a single title for this document. Only output the title, nothing else: ' + document},
        ]
        prompt = self._apply_chat_template(messages)
        preds = self._generate_prompt(prompt, max_tokens=50, temperature=0.1)
        return preds[0].outputs[0].text

    def _format_pii(self, pii):
        return json_pii_formatter(pii)

    def _generate_aggregated_info(self, piis, formatter_func):
        messages = [
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": formatter_func(piis)},
        ]
        prompt = self._apply_chat_template(messages)
        preds = self._generate_prompt(prompt, max_tokens=580, temperature=0.001)
        return extract_json_like_objects(preds[0].outputs[0].text)

    def _generate_header(self, piis):
        messages = [
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": 'Combine all of this information into an essay style header: ' + str(piis)},
        ]
        prompt = self._apply_chat_template(messages)
        preds = self._generate_prompt(prompt, max_tokens=500, temperature=0.001)
        return preds[0].outputs[0].text.split('header:')[1]

    def _create_full_essay(self, title, header, essay):
        document_text_templates = [
            '[TITLE]\n[AGGREGATED_INFO]\n[ESSAY]\n',
            '[TITLE] [AGGREGATED_INFO] [ESSAY]\n',
            '[AGGREGATED_INFO] [TITLE]\n[ESSAY]\n'
        ]
        document_style = random.choice(document_text_templates)
        return document_style.replace('[TITLE]', title).replace('[AGGREGATED_INFO]', header).replace('[ESSAY]', essay)

    def _apply_chat_template(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_prompt(self, prompt, max_tokens, temperature):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            skip_special_tokens=False,
            stop=[self.tokenizer.eos_token, "<|eot_id|>"]
        )
        return self.llm_model.generate(prompts=[prompt], sampling_params=sampling_params)
