import tqdm
import json
import torch
import torch.nn as nn
import datasets
import transformers
from datasets import Dataset
from utils import get_morph_nli_predictions, generate_morphs, nli_preds_over_morphs, constitency_filter, prepare_text, get_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score
import wandb
import pickle

##################
transformers.set_seed(1)
max_steps_per_bootstrap_iteration = 100_000
model_name = '/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model'
morph_lengths = [3, 4]
# morph_lengths = [4]
biter = 10
learning_rate = 3e-4
##################


path = 'MorphNLI/morph_model'
device = torch.device('cuda:0')
morph_tokenizer = AutoTokenizer.from_pretrained(path)
morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)

nli_model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}




snli_data   = datasets.load_dataset('snli')
snli_labels = snli_data['train'].features['label'].names

mnli_data   = datasets.load_dataset('glue', 'mnli')
mnli_labels = mnli_data['train'].features['label'].names

sick_data   = datasets.load_dataset('sick')
sick_labels = sick_data['train'].features['label'].names


snli_data_val     = snli_data['validation'].map(lambda x:            {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_ma_data_val  = mnli_data['validation_matched'].map(lambda x:    {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_mi_data_val  = mnli_data['validation_mismatched'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
sick_data_val     = sick_data['validation'].map(lambda x:            {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['sentence_A'], x['sentence_B'], 3)))}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})

transformers.logging.set_verbosity_error()


print("\n")
print("##############")
print(f"EVAL")
current_results = []
for morph_length in morph_lengths:
    for (label, nli_data) in (eval_pb := tqdm.tqdm([
        ('snli',    snli_data_val),
        ('mnli_ma', mnli_ma_data_val),
        ('mnli_mi', mnli_mi_data_val),
        ('sick',    sick_data_val),
    ])):
        eval_pb.set_description(f'S2, {label}, {morph_length}')
        dl       = torch.utils.data.DataLoader(nli_data, batch_size=16, num_workers=16)
        nli_pred = get_morph_nli_predictions(dl=dl, nli_model=nli_model.eval(), nli_tokenizer=nli_tokenizer, morph_model=morph_model.eval(), morph_tokenizer=morph_tokenizer, morph_length=morph_length, nli_id2label=nli_id2label, device=device)
        nli_gold = nli_data['label']
        score    = accuracy_score(nli_gold, nli_pred)

        current_results.append({
            'morph_length'       : morph_length,
            'dataset'            : label,
            'score'              : score,
        })
        print(f'{label}_{morph_length} -> {score}')
print("##############")
print("\n")