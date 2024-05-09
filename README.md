# Detection of Personal Identifiable Information (PII) Using Large Language Models (LLMs)

 Utilizing Large Language Models (LLMs) to detect and remove PII from educational data.

## Overview

The project focuses on developing a model to automatically detect personally identifiable information (PII) in student writings. This initiative aims to streamline the process of cleansing educational datasets from PII, thereby reducing the costs and complexities associated with making these datasets publicly available. Such advancements will facilitate research in learning sciences and aid in the creation of innovative educational tools, enhancing educational practices and methodologies.

Currently, the most reliable method to ensure that educational data does not contain PII is through manual screening, which is both time-consuming and costly. Existing automated techniques, primarily those based on named entity recognition (NER), are efficient only for detecting uniformly formatted PII like emails and phone numbers. The project addresses the need for more sophisticated systems that can accurately identify various forms of PII, distinguishing between sensitive and non-sensitive information, which is crucial for maintaining student privacy.

This project is poised to significantly impact the accessibility of educational data for research purposes. By improving PII detection, the data can be made publicly available without privacy concerns, enabling researchers and educational developers to leverage this information. This will lead to the development of effective educational tools and interventions that can support teachers and students, fostering a more enriched and effective learning environment.

## Installation

To prepare your environment and set up the project, follow these detailed steps:

1. **Clone the Repository**
   
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/jdelarosaquiros/pii_detection.git

2. **Install Python**
   
    Ensure that Python 3.11 is installed on your system. If not, you can download it from the official Python website.

3. **Create and Activate a Virtual Environment**
   
    It's recommended to use a virtual environment to manage dependencies.

4. **Install Dependencies**
   
     With the virtual environment activated, install the necessary dependencies. Start with the basic requirements from the requirements.txt file:
    ```bash
    pip install -r requirements.txt

## Usage

In our project, we focused on enhancing the performance of the LLAMA3 8B model through a comprehensive fine-tuning approach. We used a custom fine-tuning dataset tailored specifically to our needs. This dataset was crucial in training the model to differentiate and identify PII more effectively.

We incorporated clean-up functions to improve the model's reliability and reduce errors such as hallucinations. Additionally, we formatted the prompts in such a way that the model outputs could be easily transformed into structured dictionaries, mapping labels to their respective PII categories. This formatting aids in practically applying the model's outputs, making it easier to integrate and use in real-world scenarios. Below is the prompt we used for fine-tuning the model:

```bash
  f'''<|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
  
  ### Instruction:
  Your task is to identify and list any personal information in the input text. Please, list the following personal information.
  
  List of Names:
  * first full or partial name of a person
  
  List of Emails:
  * email address of a person
  
  List of Usernames:
  * username of a person
  
  List of ID Numbers:
  * number or sequence of characters that could be used to identify a person like the student ID or a social security number
  
  List of Phone Numbers:
  * phone number associated with a person
  
  List of URLs:
  * URL that might be used to identify a person
  
  List of Street Addresses:
  * full or partial street address that is associated with the person, such as a home address
  
  ### Input:
  Text: {text}
  
  ### Response:
  {answer}'''
```

Moreover, we utilized a tool called Unsloth to train our model in a quantized mode with LoRA Adapters, optimizing performance without sacrificing the model’s depth of knowledge. We created an Instruction-Based Dataset that adheres to an instruction-answering format. This dataset reconstructs sentences and their corresponding labels from tokenized inputs. It has been specifically modified from two large PII datasets originally designed for BERT, adapting them to suit Large Language Model (LLM) instructions.

Our datasets include a diverse set of sources: The Kaggle Essay Dataset, which consists of 22,000 student essays, and the Ai 4 Privacy PII Masking 300k dataset, which includes 30k English paragraphs. These datasets are labeled with various categories of PII such as names, emails, ID numbers, phone numbers, addresses, URLs, and usernames. To augment the model's general knowledge and ensure a robust instructional format, we merged these datasets with the Alpaca dataset. The final custom Instruction-Based Dataset is a synthesis of the Alpaca, Masking, and Essay datasets, amounting to 118,000 instruction samples.

This strategic integration and fine-tuning of datasets and model functionalities have been directed at significantly boosting the model's ability to process and analyze text data effectively, ensuring that it can recognize and classify PII accurately in a variety of contexts.

1. **Model**

   - LLaMa-3 (8B) [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
     
2. **Dataset**
    
   - [The Kaggle Essay Dataset](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)
   
   - 30K English paragraphs of [Ai 4 Privacy PII Masking 300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)
   
   - Custom Instruction-Based Dataset​:
     Combination of [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), Masking, and Kaggle Essay datasets​
   
3. **Run the Notebook:**
   - Open the notebook in Jupyter or another compatible environment and execute the cells sequentially.

4. **Fine-tune the Model:**
   - Fine-tune the model with our instruction-tuning datasets
   - After  fine-tuning, the model will be saved locally for further use or evaluation.

5. **Evaluate the Model:**
   - Evaluate the model on both the instruction-tuning dataset and the Kaggle essay dataset

## Performance Metrics: On the Instruction-Tuning Dataset

| Model                 | BLEU  | ROUGE-L | BERT-F1 |
|-----------------------|-------|---------|---------|
| Llama 3 8B Vanilla    | 0.470 | 0.605   | 0.931   |
| Llama 3 8B Fine-tuned | 0.627 | 0.753   | 0.978   |

The results clearly show that fine-tuning the Llama 3 8B model has significantly improved its performance across all evaluated metrics. The BLEU score, which measures how closely the model's generated text matches a reference text, increased from 0.470 in the vanilla version to 0.627 in the fine-tuned version. This indicates a better quality of generated text in terms of syntactic and semantic accuracy.

Similarly, the ROUGE-L score, which assesses the overlap of the longest common subsequence between the generated text and the reference, improved from 0.605 to 0.753. This improvement suggests that the fine-tuned model is more effective at capturing and reproducing the structure and content of the reference texts.

Lastly, the BERT-F1 score, which is a measure of the contextual similarity between the generated and reference texts, rose from 0.931 to 0.978. This significant increase reflects a more nuanced understanding and replication of the context and meaning of the input text, demonstrating the effectiveness of fine-tuning in enhancing the model's overall linguistic and contextual accuracy.

## Performance Metrics: On the Kaggle Essay Dataset
Accuracy PII only includes correct or incorrect classification of PII tokens, overall accuracy includes non-PII correct classifications​

| Model                 | Accuracy on PII | Accuracy Overall | Hallucination Rate |
|-----------------------|-----------------|------------------|--------------------|
| Pure Regex            | 0.213           | -                | -                  |
| Llama 3 8B Vanilla    | 0.862           | 0.977            | 0.620              |
| Llama 3 8B Fine-tuned | 0.967           | 0.994            | 0.432              |

**Pure Regex:** This model, which solely relies on regular expressions for PII detection, shows a low accuracy of 0.213. This indicates that while regex can catch straightforward patterns, it struggles with the complexity and variability inherent in natural language data. The overall accuracy and hallucination rate are not applicable or not measured for this model.

**Llama 3 8B Vanilla:** This version of the Llama model shows a significant improvement in accuracy on PII detection at 0.862, compared to the Pure Regex model. Its overall accuracy is quite high at 0.977, indicating strong performance in correctly processing and analyzing the data. However, the hallucination rate is 0.620, suggesting that while the model is accurate, it often generates or predicts information that isn't supported by the input data.

**Llama 3 8B Fine-tuned:** The fine-tuned version of the Llama 3 8B model exhibits the best performance across all metrics. It achieves a high accuracy on PII detection of 0.967 and an overall accuracy of 0.994, indicating excellent capability in identifying PII accurately and handling diverse data effectively. Moreover, the hallucination rate is reduced to 0.432, showing a substantial decrease in generating unsupported information compared to its vanilla counterpart.
These results demonstrate the effectiveness of fine-tuning Llama 3 8B on specialized tasks such as PII detection, significantly enhancing its precision and reliability while reducing errors like hallucinations.
