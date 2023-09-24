"""
Train an `AutoModelForSequenceClassification` for NLI
This involves:
- Setting a mapping between label and label_id
    This will be used to initialize the linear layer on top of the pre-trained transformer (we use bert-base-cased)
- Load training data
    We use a concatenation of SNLI and MNLI
- Create the config using `TrainingArguments`
- Train and Save
"""
import datasets
import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import tqdm

set_seed(1)

label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model     = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3, id2label=id2label, label2id=label2id)

snli = datasets.load_dataset('snli')
mnli = datasets.load_dataset('glue', 'mnli')


train = datasets.concatenate_datasets([snli['train'], mnli['train']]).map(lambda x: tokenizer([(premise, hypothesis) for (premise, hypothesis) in zip(x['premise'], x['hypothesis'])], truncation=True, max_length=512), batched=True).remove_columns(['premise', 'hypothesis', 'idx']).filter(lambda x: x['label'] in [0, 1, 2])
dev   = datasets.concatenate_datasets([snli['validation'], mnli['validation_matched'], mnli['validation_mismatched']]).map(lambda x: tokenizer([(premise, hypothesis) for (premise, hypothesis) in zip(x['premise'], x['hypothesis'])], truncation=True, max_length=512), batched=True).remove_columns(['premise', 'hypothesis', 'idx']).filter(lambda x: x['label'] in [0, 1, 2])



output_dir = '/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model'

training_args = TrainingArguments(
    output_dir                  = output_dir,
    fp16                        = False,
    # fp16_backend                = "amp",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    # eval_accumulation_steps     = 16,
    evaluation_strategy         = "steps",
    eval_steps                  = 5000,      #logging_steps,
    save_steps                  = 5000,
    logging_steps               = 500,
    save_total_limit            = 2,
    max_steps                   = 100_000,
    gradient_accumulation_steps = 1,
    # report_to                   = "wandb",
    remove_unused_columns       = False,
    # weight_decay                = 0.001,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = 'linear',
    dataloader_num_workers      = 16,
    learning_rate               = 1e-5,
    load_best_model_at_end      = True,
)

data_collator = DataCollatorWithPadding(
    tokenizer,
    pad_to_multiple_of=8,
)

# dl = torch.utils.data.DataLoader(train, collate_fn = data_collator, batch_size=4)
# for batch in tqdm.tqdm(dl):
#     print(model(**batch))

# exit()
# batch = next(iter(dl))
# device = torch.device('cuda:0')
# batch.to(device)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions==labels).sum() / predictions.shape[0]
    return accuracy

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train,
    eval_dataset    = dev,
    # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    # compute_metrics = compute_metrics
)

trainer.train()

trainer.save_model(output_dir)
