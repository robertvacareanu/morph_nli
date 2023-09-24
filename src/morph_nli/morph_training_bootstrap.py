"""
Train the Morph NLI Model


"""

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
biter = 50
learning_rate = 3e-4
##################



wandb.init(project="natlog", name="bootstrapping_230510")


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

snli_data_prepared = snli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True)
mnli_data_prepared = mnli_data['train'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'}).shuffle(seed=1, keep_in_memory=True)
sick_data_prepared = sick_data['train'].map(lambda x: {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})


nli_data_train = []

# Currently we store this two time in memory
nli_data_train_id_to_line = {}

for (i, line) in enumerate(datasets.concatenate_datasets([snli_data_prepared, mnli_data_prepared, sick_data_prepared])):
    current = {'id': i, **line, 'length': len(morph_tokenizer.encode(prepare_text(line['premise'], line['hypothesis'], 3)))}
    if current['length'] < 88:
        nli_data_train.append(current)
        nli_data_train_id_to_line[i] = current

nli_data_train = sorted(nli_data_train, key=lambda x: x['length'])

nli_data_train = Dataset.from_list(nli_data_train)

print(nli_data_train)

snli_data_val     = snli_data['validation'].map(lambda x:            {'p': x['premise'],    'h': x['hypothesis'], 'l': snli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_ma_data_val  = mnli_data['validation_matched'].map(lambda x:    {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
mnli_mi_data_val  = mnli_data['validation_mismatched'].map(lambda x: {'p': x['premise'],    'h': x['hypothesis'], 'l': mnli_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['premise'], x['hypothesis'], 3)))}, load_from_cache_file=False).remove_columns(['premise', 'hypothesis', 'label', 'idx']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})
sick_data_val     = sick_data['validation'].map(lambda x:            {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower(), 'length': len(morph_tokenizer.encode(prepare_text(x['sentence_A'], x['sentence_B'], 3)))}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})

transformers.logging.set_verbosity_error()


results = []
output_dir = '/data/nlp/morphing/230510/bootstrapping/'
for bootstrapping_iteration in range(biter):
    """
    Each iteration, do the following
    1. Use the previous iteration text-morphing model to annotate nli_data_train (use the one trained on 
    the data from Text Morphing paper for iteration 0) 
    2. Train text morphing model on the data from step (1) (NOTE: Maybe allow for reusing prev model) 
    3. Record NLI performance on nli_data_val
    """
    output_dir_current_iter = f'{output_dir}/models/boostrap_it_{bootstrapping_iteration}'

    # STEP 1: ANNOTATE
    nli_data_from_morphs = []
    for morph_length in morph_lengths:
        dl = torch.utils.data.DataLoader(nli_data_train.filter(lambda x: x['length'] < 264//morph_length), batch_size=64, num_workers=32)
        for batch in (pg_annotate := tqdm.tqdm(dl)):
            pg_annotate.set_description(f"S1, {bootstrapping_iteration}, {morph_length}")
            # We generate the morphs over this batch
            generated_morphings = generate_morphs(batch, morph_model, morph_tokenizer, morph_length=morph_length, device=device)
            # We obtain the NLI prediction over the morphs previously generated by using an NLI model
            nli_pred_on_morphs = nli_preds_over_morphs(generated_morphings, nli_model, nli_tokenizer, nli_id2label, device=device)
            # We filter for constitency
            nli_data_from_morphs += [{'source': batch['premise'][i], 'target': batch['hypothesis'][i], 'edits': generated_morphings[i]} for (i, x) in enumerate(nli_pred_on_morphs) if constitency_filter(x, batch['label'][i], morph_length)]

    with open(f'{output_dir}/data/data_annotated_it_{bootstrapping_iteration}.jsonl', 'w+') as fout:
        for line in nli_data_from_morphs:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

    # Reinit morph model
    morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)
    morph_model.train()
    # STEP 2: TRAIN
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir_current_iter,
        fp16                        = False,
        # fp16_backend                = "amp",
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        # eval_accumulation_steps     = 16,
        evaluation_strategy         = "steps",
        eval_steps                  = 500_000,      #logging_steps,
        save_steps                  = max_steps_per_bootstrap_iteration//2,
        logging_steps               = 500,
        save_total_limit            = 1,
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
        # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
        tokenizer       = morph_tokenizer,
        data_collator   = data_collator,
        # compute_metrics = compute_metrics
    )

    trainer.train()
    trainer.save_model(f'{output_dir}/models/final_model')

    # STEP 3: RECORD NLI PERFORMANCE
    print("\n")
    print("##############")
    print(f"EVAL: {bootstrapping_iteration}")
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

            wandb.log({f'morph_length{morph_length}/{label}': score, 'step': bootstrapping_iteration})

            current_results.append({
                'bootstrap_iteration': bootstrapping_iteration,
                'morph_length'       : morph_length,
                'dataset'            : label,
                'score'              : score,
            })
            print(f'{label}_{morph_length} -> {score}')
    print("##############")
    print("\n")
    with open(f'{output_dir}/models/boostrap_results_endof_it_{bootstrapping_iteration}.pkl', 'wb+') as fout:
        pickle.dump(current_results, fout)
    results += current_results


with open(f'{output_dir}/models/boostrap_results.pkl', 'wb+') as fout:
    pickle.dump(results, fout)

wandb.finish()