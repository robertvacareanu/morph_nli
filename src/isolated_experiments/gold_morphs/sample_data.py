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

def get_snli_data():
    dataset_name = 'snli'
    snli = datasets.load_dataset('snli')['validation']
    features = snli.features['label'].names
    return (dataset_name, snli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_matched_data():
    dataset_name = 'glue_mnli_matched'
    mnli     = datasets.load_dataset('glue', 'mnli')['validation_matched']
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_mismatched_data():
    dataset_name = 'glue_mnli_mismatched'
    mnli     = datasets.load_dataset('glue', 'mnli')['validation_mismatched']
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_sick_data():
    dataset_name = 'sick'
    sick     = datasets.load_dataset(dataset_name)['validation']
    features = sick.features['label'].names
    return (dataset_name, sick, features, 'sentence_A', 'sentence_B', lambda pred: pred)


device = torch.device('cuda:0')

nli_model = AutoModelForSequenceClassification.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model").to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model")

nli_id2label = nli_model.config.id2label

for (dataset_name, nli_data, features, premise_column, hypothesis_column, post_process) in [
    # get_snli_data(),  
    # get_mnli_matched_data(), 
    # get_mnli_mismatched_data(), 
    get_sick_data(), 
]:
    
    # Get predictions

    dl = torch.utils.data.DataLoader(nli_data, batch_size=20)

    # nli_id2label[nli_model(**nli_tokenizer([(e, c)], return_tensors='pt')).logits[0].argmax().detach().cpu().numpy().item()]
    pred = []
    text = []
    gold = []
    for batch in tqdm.tqdm(dl):
        tokenized = nli_tokenizer([(e, c) for (e, c) in zip(batch[premise_column], batch[hypothesis_column])], return_tensors='pt', padding=True, max_length=512).to(device)
        output    = nli_model(**tokenized).logits.argmax(dim=1).detach().cpu().tolist()
        pred += [nli_id2label[x].lower() for x in output]
        gold += [features[x].lower() for x in batch['label'].tolist()]
        text += [{'premise': x, 'hypothesis': y} for (x, y) in zip(batch[premise_column], batch[hypothesis_column])]

    # Split
    groups = defaultdict(list)
    for p, g, t in zip(pred, gold, text):
        groups[('correct' if p==g else 'incorrect', g)].append({**t, 'gold': g, 'predicted': p, 'morphs': []})
    
    # Select
    for key in groups.keys():
        # We make the samples agnostic to the order of the keys
        set_seed(1)
        samples = random.choices(groups[key], k=50)
    
        Path(f'{dataset_name}_data').mkdir(exist_ok=True)
        with open(f'{dataset_name}_data/{key[0]}_{key[1]}.jsonl', 'w+') as fout:
            for line in samples:
                _=fout.write(json.dumps(line))
                _=fout.write('\n') 