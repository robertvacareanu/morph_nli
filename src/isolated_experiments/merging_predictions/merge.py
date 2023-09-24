import tqdm
import json
import torch
import torch.nn as nn
import datasets
import transformers
from datasets import Dataset
from src.morph_nli.utils import get_morph_nli_predictions, get_vanilla_nli_predictions, generate_morphs, nli_preds_over_morphs, constitency_filter, prepare_text, get_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
import wandb
import pickle

##################
transformers.set_seed(1)
max_steps_per_bootstrap_iteration = 50_000
model_name = '/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model'
morph_lengths = [3]
# morph_lengths = [4]
biter = 50
learning_rate = 3e-4
##################

device = torch.device('cuda:0')

nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
nli_tokenizer = AutoTokenizer.from_pretrained(model_name)

nli_id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


# path = 'MorphNLI/morph_model'
path = '/data/nlp/morphing/230510/bootstrapping/models/boostrap_it_37/checkpoint-50000/'
morph_tokenizer = AutoTokenizer.from_pretrained(path)
morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)


snli_data   = datasets.load_dataset('snli')
snli_labels = snli_data['train'].features['label'].names

mnli_data   = datasets.load_dataset('glue', 'mnli')
mnli_labels = mnli_data['train'].features['label'].names

sick_data   = datasets.load_dataset('sick')
sick_labels = sick_data['train'].features['label'].names

snli_data_prepared = snli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True)
mnli_data_prepared = mnli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True)
sick_data_prepared = sick_data['train'].map(lambda x: {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})


snli_data_val     = snli_data['validation'].map(lambda x:            {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_ma_data_val  = mnli_data['validation_matched'].map(lambda x:    {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_mi_data_val  = mnli_data['validation_mismatched'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
sick_data_val     = sick_data['validation'].map(lambda x:            {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['sentence_A'], x['sentence_B'], 3)))}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})


print("\n")
print("##############")
current_results = []
(label, nli_data) = ('snli',    snli_data_val)

dl       = torch.utils.data.DataLoader(nli_data, batch_size=16, num_workers=16)
nli_pred = get_morph_nli_predictions(dl=dl, nli_model=nli_model.eval(), nli_tokenizer=nli_tokenizer, morph_model=morph_model.eval(), morph_tokenizer=morph_tokenizer, morph_length=3, nli_id2label=nli_id2label, device=device)
van_pred = get_vanilla_nli_predictions(dl=dl, nli_model=nli_model.eval(), nli_tokenizer=nli_tokenizer, nli_id2label=nli_id2label, device=device)

premises   = [x['premise'] for x in nli_data]
hypothesis = [x['hypothesis'] for x in nli_data]

from nltk.metrics import edit_distance
from sentence_transformers import SentenceTransformer, util
import pandas as pd
edistances = [edit_distance(x, y) for (x, y) in tqdm.tqdm(zip(premises, hypothesis), total=len(premises))]



embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_distances = [util.pytorch_cos_sim(embedding_model.encode(p, convert_to_tensor=True), embedding_model.encode(h, convert_to_tensor=True)).item() for (p, h) in tqdm.tqdm(zip(premises, hypothesis), total=len(premises))]

nli_gold   = nli_data['label']

print(accuracy_score(nli_gold, nli_pred))
print(accuracy_score(nli_gold, van_pred))

accuracy_score(nli_gold, [m if (min(len(prem.split(' ')), len(hypo.split(' '))) / max(len(prem.split(' ')), len(hypo.split(' ')))) < 0.2 else v for (m, v, prem, hypo) in zip(nli_pred, van_pred, premises, hypothesis)])







df = pd.DataFrame([{'edit_distance': d, 'length_distance': (min(len(prem.split(' ')), len(hypo.split(' '))) / max(len(prem.split(' ')), len(hypo.split(' ')))), 'emb_distance': emb, 'morph': 1 if m==g else 0, 'vanilla': 1 if v==g else 0} for (d, m, v, g, emb, prem, hypo) in zip(edistances, nli_pred, van_pred, nli_gold, embedding_distances, premises, hypothesis)])


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x='length_distance', hue='morph')
plt.yscale('log')
plt.savefig('/home/rvacareanu/projects_7_2309/morph_nli/src/isolated_experiments/merging_predictions/hist_morph3.png')
plt.clf()
plt.cla()
sns.histplot(data=df, x='length_distance', hue='vanilla')
plt.yscale('log')
plt.savefig('/home/rvacareanu/projects_7_2309/morph_nli/src/isolated_experiments/merging_predictions/hist_vanilla3.png')
plt.clf()
plt.cla()

# sns.histplot(data=df, x='emb_distance', hue='morph')
# plt.yscale('log')
# plt.savefig('/home/rvacareanu/projects_7_2309/morph_nli/src/isolated_experiments/merging_predictions/hist_morph2.png')
# plt.clf()
# plt.cla()
# sns.histplot(data=df, x='emb_distance', hue='vanilla')
# plt.yscale('log')
# plt.savefig('/home/rvacareanu/projects_7_2309/morph_nli/src/isolated_experiments/merging_predictions/hist_vanilla2.png')
# plt.clf()
# plt.cla()
