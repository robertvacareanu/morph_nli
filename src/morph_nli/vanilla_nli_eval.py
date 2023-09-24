import tqdm
import torch
import torch.nn as nn
import datasets
import random
from transformers import RobertaTokenizer, AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSequenceClassification, set_seed
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

set_seed(1)

def get_snli_data():
    dataset_name = 'snli'
    snli = datasets.load_dataset('snli')['validation']
    features = snli.features['label'].names
    return (dataset_name, snli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_matched_data():
    dataset_name = 'glue/mnli/matched'
    mnli     = datasets.load_dataset('glue', 'mnli')['validation_matched']
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_mismatched_data():
    dataset_name = 'glue/mnli/mismatched'
    mnli     = datasets.load_dataset('glue', 'mnli')['validation_mismatched']
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_anli_data():
    dataset_name = 'anli'
    anli     = datasets.load_dataset(dataset_name)['dev_r1']
    features = anli.features['label'].names
    return (dataset_name, anli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_sick_data():
    dataset_name = 'sick'
    sick     = datasets.load_dataset(dataset_name)['validation']
    features = sick.features['label'].names
    return (dataset_name, sick, features, 'sentence_A', 'sentence_B', lambda pred: pred)

def get_hans_data():
    dataset_name = 'hans'
    hans     = datasets.load_dataset(dataset_name)['validation']
    features = hans.features['label'].names
    return (dataset_name, hans, features, 'premise', 'hypothesis', lambda pred: ['entailment' if x == 'entailment' else 'non-entailment' for x in pred])



device = torch.device('cuda:0')

nli_model = AutoModelForSequenceClassification.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model").to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("/home/rvacareanu/projects_6_2301/alignment/natlog/results/230502/vanilla_nli/model")

nli_id2label = nli_model.config.id2label


for (dataset_name, nli_data, features, premise_column, hypothesis_column, post_process) in [
    get_snli_data(),  
    get_mnli_matched_data(), 
    get_mnli_mismatched_data(), 
    get_anli_data(), 
    get_sick_data(), 
    get_hans_data(), 
]:
    random.seed(1)

    dl = torch.utils.data.DataLoader(nli_data, batch_size=20)

    # nli_id2label[nli_model(**nli_tokenizer([(e, c)], return_tensors='pt')).logits[0].argmax().detach().cpu().numpy().item()]
    pred1 = []
    text  = []
    gold = []
    for batch in tqdm.tqdm(dl):
        tokenized = nli_tokenizer([(e, c) for (e, c) in zip(batch[premise_column], batch[hypothesis_column])], return_tensors='pt', padding=True, max_length=512).to(device)
        output    = nli_model(**tokenized).logits.argmax(dim=1).detach().cpu().tolist()
        pred1 += [nli_id2label[x].lower() for x in output]
        gold += [features[x].lower() for x in batch['label'].tolist()]
        text += list(zip(batch[premise_column], batch[hypothesis_column]))

    pred1 = post_process(pred1)
    
    print("############")
    print(dataset_name)
    print(accuracy_score(gold, pred1))
    print(confusion_matrix(gold, pred1, labels=sorted(list(set(gold)))))
    print(sorted(Counter(zip(gold, pred1)).items(), key=lambda x: x[0]))
    print(sorted(Counter(pred1).items(), key=lambda x: x[0]))
    print(sorted(Counter(gold).items(), key=lambda x: x[0]))
    wrong = [(i, p, g) for (i, (p, g)) in enumerate(zip(pred1, gold)) if p != g]
    labels = sorted(list(set(gold)))
    print({l: [(x[0], x[1]) for x in wrong if x[2] == l][:10] for l in labels})
    print({l: random.choices([(x[0], x[1]) for x in wrong if x[2] == l], k=10) if len([(x[0], x[1]) for x in wrong if x[2] == l]) > 10 else [(x[0], x[1]) for x in wrong if x[2] == l] for l in labels})
    print("############")







