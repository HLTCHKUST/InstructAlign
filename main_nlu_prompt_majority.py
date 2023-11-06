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
import random

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from nusacrowd import NusantaraConfigHelper
from collections import Counter

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

if __name__ == '__main__':
    prompt_lang = 'EN' # DUMMY

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

        
        # preprocess label (lower case & translate)
        label_names = [str(label).lower().replace("_"," ") for label in label_names]
        labels += label_names
        
        majority = Counter(data['label']).most_common(1)[0][0]
        
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
        with torch.inference_mode():
            for sample in tqdm(data):
                inputs.append(sample['text'])
                preds.append(label_to_id_dict[majority] if type(majority) == str else majority)
                golds.append(label_to_id_dict[sample['label']] if type(sample['label']) == str else sample['label'])

        inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
        inference_df.to_csv(f'outputs/{dset_subset}_{prompt_lang}_majority.csv', index=False)

        acc, f1 = accuracy_score(golds, preds), f1_score(golds, preds, average='macro')
        print(dset_subset)
        print('accuracy', acc)
        print('f1 macro', f1)
        metrics[dset_subset] = {'accuracy': acc, 'f1_score': f1}
        print("===\n\n")

        pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics/nlu_results_{prompt_lang}_majority.csv', index=False)