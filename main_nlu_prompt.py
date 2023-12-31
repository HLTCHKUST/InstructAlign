"""nusacrowd zero-shot prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ru8DyS2ALWfRdkjOPHj-KNjw6Pfa44Nd
"""
import os, sys
import csv
from os.path import exists

from numpy import argmax
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from nlu_prompt import get_prompt

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from nusacrowd import NusantaraConfigHelper

from data_utils import load_xnli_dataset, load_nusa_menulis_dataset, load_nlu_tasks
#!pip install git+https://github.com/IndoNLP/nusa-crowd.git@release_exp
#!pip install transformers
#!pip install sentencepiece

DEBUG=False

def to_prompt(input, prompt, labels, prompt_lang):
    # single label
    if 'text' in input:
        prompt = prompt.replace('[INPUT]', input['text'])
    else:
        prompt = prompt.replace('[INPUT_A]', input['text_1'])
        prompt = prompt.replace('[INPUT_B]', input['text_2'])

    # replace [OPTIONS] to A, B, or C
    if "[OPTIONS]" in prompt:
        new_labels = [f'{l}' for l in labels]
        new_labels[-1] = ("or " if 'EN' in prompt_lang else  "atau ") + new_labels[-1] 
        if len(new_labels) > 2:
            prompt = prompt.replace('[OPTIONS]', ', '.join(new_labels))
        else:
            prompt = prompt.replace('[OPTIONS]', ' '.join(new_labels))

    return prompt


@torch.no_grad()
def get_logprobs(model, tokenizer, prompt, label_ids=None, label_attn=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
        return logprobs.sum() / label_attn.sum()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
        return logprobs.mean()

def predict_classification(model, tokenizer, prompt, labels):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids =labels_encoded['input_ids'].to('cuda')
        list_label_attn =labels_encoded['attention_mask'].to('cuda')
        probs = [
                    get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', ''), label_ids.view(1,-1), label_attn.view(1,-1)) 
                     for (label_ids, label_attn) in zip(list_label_ids, list_label_attn)
                ]
    else:
        probs = [get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', label)) for label in labels]
    return probs

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('main_nlu_prompt.py <prompt_lang> <model_path_or_name> <optional_output_name>')

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]

    output_name = None
    if len(sys.argv) == 4:
        output_name = sys.argv[3]
        
    os.makedirs('./outputs', exist_ok=True) 

    # Load Prompt
    DATA_TO_PROMPT = get_prompt(prompt_lang)

    # Load Dataset
    print('Load NLU Datasets...')
    nlu_datasets = load_nlu_tasks()
    nusa_menulis_dataset = load_nusa_menulis_dataset()
    # xnli_dataset = load_xnli_dataset()

    nlu_datasets.update(nusa_menulis_dataset)
    # nlu_datasets.update(xnli_dataset)

    print(f'Loaded {len(nlu_datasets)} NLU datasets')
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side='left')
    if "bloom" in MODEL or "xglm" in MODEL or "gpt2" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
        tokenizer.pad_token = tokenizer.eos_token # Use EOS to pad label
        
    model.eval()
    torch.no_grad()

    metrics = {}
    labels = []
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')
        if dset_subset not in DATA_TO_PROMPT or DATA_TO_PROMPT[dset_subset] is None:
            print('SKIP')
            continue

        if 'test' in nlu_datasets[dset_subset]:
            data = nlu_datasets[dset_subset]['test']
        else:
            data = nlu_datasets[dset_subset]['train']

        if DEBUG:
            print(dset_subset)

        try:
            label_names = data.features['label'].names
        except:
            label_names = list(set(data['label']))
        label_to_id_dict = { l : i for i, l in enumerate(label_names) }

        # normalize some labels for more natural prompt:
        if dset_subset == 'imdb_jv_nusantara_text':
            label_names = ['positive', 'negative']
        elif dset_subset == 'indonli_nusantara_pairs':
            label_names = ['no', 'yes', 'maybe']
        elif 'xnli' in dset_subset:
            xnli_map = {'neutral': 'inconclusive', 'contradiction': 'false', 'entailment': 'true'}
            label_names = list(map(lambda x: xnli_map[x], label_names))

        en_id_label_map = {
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',	'special': 'khusus', 'general': 'umum',
            'no': 'tidak', 'yes': 'ya', 'maybe': 'mungkin', 'negative': 'negatif', 'positive': 'positif', 
            'east': 'timur', 'standard': 'standar', 'ngapak': 'ngapak', 'unknown': 'unknown',
            'neutral': 'netral', 'love': 'cinta', 'fear': 'takut', 'happy': 'senang', 'sad': 'sedih',
            'sadness': 'sedih', 'disgust': 'jijik', 'anger': 'marah', 'surprise': 'terkejut', 'joy': 'senang',
            'reject': 'ditolak', 'tax': 'pajak', 'partial': 'sebagian', 'others': 'lain-lain',
            'granted': 'dikabulkan', 'fulfill': 'penuh', 'correction': 'koreksi',
            'not abusive': 'tidak abusive', 'abusive': 'abusive', 'abusive and offensive': 'abusive dan offensive',
            'support': 'mendukung', 'against': 'bertentangan', 
        }
        
        # preprocess label (lower case & translate)
        label_names = [str(label).lower().replace("_"," ") for label in label_names]
        labels += label_names
        
        if 'ID' in prompt_lang:
            label_names = list(map(lambda lab: en_id_label_map[lab], label_names))

        # sample prompt
        print("LABEL NAME = ")
        print(label_names)
        print("SAMPLE PROMPT = ")
        print(to_prompt(data[0], DATA_TO_PROMPT[dset_subset], label_names, prompt_lang))
        print("\n")

        inputs = []
        preds = []
        golds = [] 
        # zero-shot inference
        if not exists(f'outputs/{dset_subset}_{prompt_lang}_{MODEL.split("/")[-1]}.csv'):
            with torch.inference_mode():
                for sample in tqdm(data):
                    prompt_text = to_prompt(sample, DATA_TO_PROMPT[dset_subset], label_names, prompt_lang)
                    out = predict_classification(model, tokenizer, prompt_text, label_names)
                    pred = argmax([o.cpu().detach() for o in out])
                    inputs.append(prompt_text)
                    preds.append(pred)
                    golds.append(label_to_id_dict[sample['label']] if type(sample['label']) == str else sample['label'])
     
            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
            if output_name is not None:
                inference_df.to_csv(f'outputs/{dset_subset}_{prompt_lang}_{output_name}.csv', index=False)
            else:
                inference_df.to_csv(f'outputs/{dset_subset}_{prompt_lang}_{MODEL.split("/")[-1]}.csv', index=False)
        # if output log exists, skip
        else:
            print("Output exist, use existing log instead")
            with open(f'outputs/{dset_subset}_{prompt_lang}_{MODEL.split("/")[-1]}.csv') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    inputs.append(row["Input"])
                    preds.append(row["Pred"])
                    golds.append(row["Gold"])

        acc, f1 = accuracy_score(golds, preds), f1_score(golds, preds, average='macro')
        print(dset_subset)
        print('accuracy', acc)
        print('f1 macro', f1)
        metrics[dset_subset] = {'accuracy': acc, 'f1_score': f1}
        print("===\n\n")

    if output_name is not None:
        pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics/nlu_results_{prompt_lang}_{output_name}.csv', index=False)
    else:
        pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv', index=False)