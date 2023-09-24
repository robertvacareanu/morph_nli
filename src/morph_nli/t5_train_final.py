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
max_steps_per_bootstrap_iteration = 50_000
model_name = '/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model'
morph_lengths = [3, 4]
# morph_lengths = [4]
biter = 10
learning_rate = 3e-4
##################



wandb.init(project="natlog", name="bootstrapping_230501_final")


device = torch.device('cuda:0')

nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
nli_tokenizer = AutoTokenizer.from_pretrained(model_name)

nli_id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


path = 'MorphNLI/morph_model'
morph_tokenizer = AutoTokenizer.from_pretrained(path)
morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)



snli_data   = datasets.load_dataset('snli')
snli_labels = snli_data['train'].features['label'].names

mnli_data   = datasets.load_dataset('glue', 'mnli')
mnli_labels = mnli_data['train'].features['label'].names

sick_data   = datasets.load_dataset('sick')
sick_labels = sick_data['train'].features['label'].names

snli_data_prepared = snli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True).select(range(250_000), keep_in_memory=True)
mnli_data_prepared = mnli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True).select(range(250_000), keep_in_memory=True)
sick_data_prepared = sick_data['train'].map(lambda x: {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})


nli_data_from_morphs = []
edits_set = set()
for it in range(49, 50):
    with open(f'/data/nlp/morphing/230510/bootstrapping/data/data_annotated_it_{it}.jsonl') as fin:
        for line in fin:
            loaded_line = json.loads(line)
            if max(len(loaded_line['source'].split()), len(loaded_line['target'].split())) < 5:
                continue
            else:
                if ' '.join(loaded_line['edits']) not in edits_set:
                    edits_set.add(' '.join(loaded_line['edits']))
                    nli_data_from_morphs.append(loaded_line)


snli_data_val     = snli_data['validation'].map(lambda x:            {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).map(lambda x: {'dataset_number': 1})
mnli_ma_data_val  = mnli_data['validation_matched'].map(lambda x:    {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).map(lambda x: {'dataset_number': 2})
mnli_mi_data_val  = mnli_data['validation_mismatched'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).map(lambda x: {'dataset_number': 3})
sick_data_val     = sick_data['validation'].map(lambda x:            {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['sentence_A'], x['sentence_B'], 3)))}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).map(lambda x: {'dataset_number': 4})

eval_datasets = datasets.concatenate_datasets([snli_data_val, mnli_ma_data_val, mnli_mi_data_val, sick_data_val])

transformers.logging.set_verbosity_error()

morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)
morph_model.train()
# STEP 2: TRAIN
training_args = Seq2SeqTrainingArguments(
    output_dir                  = "/data/nlp/morphing/230510/bootstrapping/experiments_at_the_end/",
    fp16                        = False,
    # fp16_backend                = "amp",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    # eval_accumulation_steps     = 16,
    evaluation_strategy         = "steps",
    eval_steps                  = 1_000_000,      #logging_steps,
    save_steps                  = 5_000,
    logging_steps               = 500,
    save_total_limit            = 10,
    max_steps                   = max_steps_per_bootstrap_iteration,
    gradient_accumulation_steps = 1,
    report_to                   = "none",
    remove_unused_columns       = False,
    # weight_decay                = 0.001,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = 'linear',
    dataloader_num_workers      = 16,
    learning_rate               = learning_rate,
    # load_best_model_at_end      = True,
    label_names                 = ['labels']
)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    morph_tokenizer,
    model=morph_model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)


trainer = Seq2SeqTrainer(
    model           = morph_model,
    args            = training_args,
    train_dataset   = Dataset.from_list(nli_data_from_morphs).map(lambda x: get_jsonl(x['source'], x['target'], x['edits']), load_from_cache_file=False).map(lambda x: {**morph_tokenizer(x['input'], truncation=True, max_length=512), 'labels': morph_tokenizer(x['output'], truncation=True, max_length=512)['input_ids']}, batched=True, load_from_cache_file=False).remove_columns(['input', 'output', 'source', 'target', 'edits']).filter(lambda x: len(x['input_ids']) < 512, load_from_cache_file=False).filter(lambda x: len(x['labels']) < 512, load_from_cache_file=False),
    # eval_dataset    = eval_datasets,
    tokenizer       = morph_tokenizer,
    data_collator   = data_collator,
    # compute_metrics = compute_metrics
)

trainer.train()
trainer.save_model('"/home/rvacareanu/projects_6_2301/alignment/natlog/results/230501/bootstrapping/models/final_concat/"')

# STEP 3: RECORD NLI PERFORMANCE
print("\n")
print("##############")
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
        print(f'{label}_{morph_length} -> {score}')
print("##############")
print("\n")