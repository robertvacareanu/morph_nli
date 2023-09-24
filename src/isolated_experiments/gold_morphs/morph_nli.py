"""
Load the data we manually annotated, then compute the final label using the morph logic
"""

import tqdm
import json
import torch
import torch.nn as nn
import datasets
import random
from transformers import RobertaTokenizer, AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSequenceClassification, set_seed
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter, defaultdict
from pathlib import Path

set_seed(1)

data = []

# for data_type in ['incorrect_contradiction', 'incorrect_entailment', 'incorrect_neutral', 'correct_contradiction', 'correct_entailment', 'correct_neutral']:
# for data_type in ['incorrect_contradiction', 'incorrect_entailment', 'incorrect_neutral']:
for data_type in ['correct_contradiction', 'correct_entailment', 'correct_neutral']:
    with open(f"src/isolated_experiments/gold_morphs/snli_data/{data_type}.jsonl") as fin:
        for line in fin:
            data.append(json.loads(line))


data = [{**x, 'morphs': [x['premise'], *x['morphs'], x['hypothesis']]} for x in data]

device = torch.device('cuda:0')

nli_model = AutoModelForSequenceClassification.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model").to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model")

nli_id2label = nli_model.config.id2label

gold = [x['gold'] for x in data]
pred = []

for line in data:
    current_preds = []
    for i in range(len(line['morphs']) - 1):
        tokenized = nli_tokenizer([(line['morphs'][i], line['morphs'][i+1])], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        output    = nli_model(**tokenized).logits.argmax(dim=1)
        output    = [x.detach().detach().cpu().tolist() for x in output][0]
        current_preds.append(nli_id2label[output].lower())
    if 'neutral' in current_preds or 'contradiction' in current_preds:
        pred.append(next((x for x in current_preds if x != 'entailment')))
    else:
        pred.append('entailment')

print(accuracy_score(gold, pred))
print(accuracy_score(gold, [x['predicted'] for x in data]))
# print(Counter([]))
# Exp for explainability: 
# Cand predicted label cu vanilla NLI e corect -> evaluare cu 50 fiecare; Comparatie cu Chain-of-Thought
# Exp cu explicatii: Propozitii intermediare + label produs de sistem (daca modelul e corect, e explicatia corecta?); Populare la masked entity (same entity for a data point (prem, intermediate1, .., intermediaten, hypo))