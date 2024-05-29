
import spacy
import re
nlp = spacy.load("en_core_web_sm")

def split_into_sentences_spacy(paragraph):
    # Process the paragraph with spaCy
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def split_word_count(paragraph, char_count=50):
    sentences = split_into_sentences_spacy(paragraph)
    split_paragraphs = []
    temp_paragraph = ''

    for sentence in sentences:
        if len(temp_paragraph +' '+ sentence) < char_count:
            if temp_paragraph == '':
                temp_paragraph = sentence
            else:
                temp_paragraph += ' '+sentence
        else:
            if temp_paragraph != '':
                split_paragraphs.append(temp_paragraph)
            temp_paragraph = sentence
    if temp_paragraph != '':
        split_paragraphs.append(temp_paragraph)
    return split_paragraphs

def pii_extract(pii_string):
    # Regex pattern
    pattern = r'"([^"]*)"\s*\(([^)]*)\)'

    matches = re.findall(pattern, pii_string)

    # Extracted contents
    quotation_content = matches[0][0] if matches else None
    parenthesis_content = matches[0][1] if matches else None
    return quotation_content, parenthesis_content

def json_pii_formatter(pii_list):
    json_style = []
    for pii in pii_list:
        # Find matches
        quo, par = pii_extract(pii)
        json_style.append({'pii':quo,'type':par,'reason':''})
    return json_style

def extract_json_like_objects(text):
    segments = re.split(r'[\[\]]', text)
    segments = [segment.strip() for segment in segments if segment.strip()]

    for seg in segments:
        if '{' in seg and '}' in seg:
            return '['+seg+']'

    return []
