import datasets
import json
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer, AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model     = T5ForConditionalGeneration.from_pretrained('t5-base')

def get_jsonl(line: str):
    result = None
    split  = line.split('\t')
    source = split[0]
    target = split[-1]
    result = {
        'input' : f"""Transform from "{source}" to "{target}" in {len(split)} steps.""",
        'output': " [SSEP] ".join(split[1:-1])
    }
    return result

# train = []
# with open('/data/nlp/morphing/230923/data/text_morph_relexicalized/train.txt') as fin:
#     for line in fin:
#         train.append(get_jsonl(line))

# dev = []
# with open('/data/nlp/morphing/230923/data/text_morph_relexicalized/dev.txt') as fin:
#     for line in fin:
#         dev.append(get_jsonl(line))

# test_1 = []
# with open('/data/nlp/morphing/230923/data/text_morph_relexicalized/test_1.txt') as fin:
#     for line in fin:
#         test_1.append(get_jsonl(line))

# test_2 = []
# with open('/data/nlp/morphing/230923/data/text_morph_relexicalized/test_2.txt') as fin:
#     for line in fin:
#         test_2.append(get_jsonl(line))

# dataset = DatasetDict(
#     {
#         'train'     : Dataset.from_list(train),
#         'validation': Dataset.from_list(dev),
#         'test_1'    : Dataset.from_list(test_1),
#         'test_2'    : Dataset.from_list(test_2),
#     }
# ).filter(lambda x: not any(skip in x['output'] for skip in ['<WORK_OF_ART>', '<EVENT>', '<LAW>', '<LANGUAGE>', '<FAC>', '<PERCENT>', '<QUANTITY>', '<PRODUCT>', '<ORG>', '<PER>', '<GPE>', '<NORP>', '<ORDINAL>', '<PERSON>', '<CARDINAL>', '<DATE>', '<MONEY>', '<TIME>', '<LOC>']))


# dataset = dataset.map(lambda x: {**tokenizer(x['input'], truncation=True, max_length=512), 'labels': tokenizer(x['output'], truncation=True, max_length=512)['input_ids']}, batched=True).remove_columns(['input', 'output']).filter(lambda x: len(x['input_ids']) < 512).filter(lambda x: len(x['labels']) < 512)



# dataset.save_to_disk('/data/nlp/morphing/230923/text_morphing_initial_data/data/relexicalized/')
# dataset = datasets.load_from_disk('/home/rvacareanu/projects_6_2301/alignment/natlog/results/230501/text_morphing_initial_data/data/text_morphing_dataset')
# dataset = datasets.load_from_disk('/data/nlp/morphing/230501/text_morphing_initial_data/data/text_morphing_dataset_relexicalized/')
dataset = datasets.load_from_disk('/data/nlp/morphing/230923/text_morphing_initial_data/data/relexicalized/')

print(dataset)


output_dir = '/data/nlp/morphing/230923/text_morphing_initial_data/model'

training_args = Seq2SeqTrainingArguments(
    output_dir                  = output_dir,
    fp16                        = False,
    # fp16_backend                = "amp",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    # eval_accumulation_steps     = 16,
    evaluation_strategy         = "steps",
    eval_steps                  = 25000,      #logging_steps,
    save_steps                  = 25000,
    logging_steps               = 500,
    save_total_limit            = 2,
    max_steps                   = 50_000,
    gradient_accumulation_steps = 16,
    # report_to                   = "wandb",
    remove_unused_columns       = False,
    # weight_decay                = 0.001,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = 'linear',
    dataloader_num_workers      = 16,
    learning_rate               = 3e-4,
    # load_best_model_at_end      = True,
    label_names                 = ['labels']
)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = dataset['train'],
    eval_dataset    = dataset['validation'],
    # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    # compute_metrics = compute_metrics
)

trainer.train()
