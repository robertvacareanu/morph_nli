import tqdm
import torch
from collections import Counter


def prepare_text(src, trgt, steps):
    """
    Transform `src` to `trgt` in a number of `steps` steps
    :param src   (str)
        The source text as a string. This is the text
        we start at, the text that we wish to transform
    :param trgt  (str)
        This is the target text as a string. This is where we want 
        to arrive after a sequence of transformations
    :param steps (int)
        In how many steps to do the transformation
    """
    return f"""Transform from "{src}" to "{trgt}" in {steps} steps."""

def get_jsonl(source, target, edits):
    """
    Get a dictionary containing `input` and `output`, to be used in the
    pipeline
    :param src   (str)
        The source text as a string. This is the text
        we start at, the text that we wish to transform
    :param trgt  (str)
        This is the target text as a string. This is where we want 
        to arrive after a sequence of transformations
    :param edits (List[str])
        This is the sequence of edits, containing `source` and `target` as well
        In other words, it is a list that looks like this:
        >>> `[source, morph_1, .., morph_n, target]`
        The resulting value in the `output` field will not contain source and target
    """
    result = {
        'input' : prepare_text(source, target, len(edits)),
        'output': " [SSEP] ".join(edits[1:-1])
    }
    return result

@torch.inference_mode()
def get_morph_nli_predictions(dl, nli_model, nli_tokenizer, morph_model, morph_tokenizer, morph_length, nli_id2label, device):
    """
    Get the predictions according to the Morph NLI
    A different function is needed because the Morph NLI operates as follows:
    1. For each (premise, hypothesis) pair, generate morphing from `premise` to `hypothesis` in `morph_length` steps
    2. Record every generated morphing: [`premise`, `morph_1`, .., `morph_n`, `hypothesis`], then unroll the dataset to form
    the pairs like so:
        (premise, morph_1)
        (morph_1, morph_2)
        ...
        (morph_n, hypothesis)
    3. Run the NLI model over each resulting pair, recording the label (e.g. 'Entailment', 'Contradiction', etc)
    4. For each initial (premise, hypothesis) pair, construct the sequence of resulting labels
        For example, ['Entailment', 'Entailment', 'Contradiction']
    5. Generate the prediction based on the sequence above (e.g. for the one above, result is `Contradiction`)
    """
    nli_model.to(device)
    morph_model.to(device)

    pred = []
    for batch in tqdm.tqdm(dl):
        prepared_text_for_morphing = [prepare_text(src, trgt, morph_length) for (src, trgt) in zip(batch['premise'], batch['hypothesis'])]
        morphed_sequences          = morph_model.generate(**morph_tokenizer(prepared_text_for_morphing, return_tensors='pt', padding='longest', truncation=True, max_length=512).to(device), max_length=(morph_length - 2) * max(batch['length']))
        morphed_sequences_decoded  = [[src] + x.split(" [SSEP] ") + [target] for (src, x, target) in zip(batch['premise'], morph_tokenizer.batch_decode(morphed_sequences, skip_special_tokens=True), batch['hypothesis'])]

        # We cannot just append <morph_length> because sometimes the text morphing model fails to produce valid outputs
        morphed_lengths            = [len(x) for x in morphed_sequences_decoded]

        nli_model_input = []
        for x in morphed_sequences_decoded:
            for i in range(len(x) - 1):
                nli_model_input.append((x[i], x[i+1]))
        

        tokenized = nli_tokenizer(nli_model_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        output    = nli_model(**tokenized).logits.argmax(dim=1).split([x-1 for x in morphed_lengths])
        output    = [x.detach().detach().cpu().tolist() for x in output]
        output    = [[nli_id2label[y].lower() for y in x] for x in output]
        for o in output:
            if all(x == 'entailment' for x in o):
                pred.append('entailment')
            else:
                pred.append(next((x for x in o if x != 'entailment'), None))

    return pred

@torch.inference_mode()
def get_vanilla_nli_predictions(dl, nli_model, nli_tokenizer, nli_id2label, device):
    """
    
    """
    nli_model.to(device)

    pred = []

    for batch in tqdm.tqdm(dl):
        tokenized = nli_tokenizer([(e, c) for (e, c) in zip(batch['premise'], batch['hypothesis'])], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        output    = nli_model(**tokenized).logits.argmax(dim=1).detach().cpu().tolist()
        pred     += [nli_id2label[x].lower() for x in output]

    return pred

@torch.inference_mode()
def generate_morphs(batch, morph_model, morph_tokenizer, morph_length, device):
    """
    Generate morphs for this batch
    """
    prepared_text_for_morphing = [prepare_text(src, trgt, morph_length) for (src, trgt) in zip(batch['premise'], batch['hypothesis'])]
    morphed_sequences          = morph_model.generate(**morph_tokenizer(prepared_text_for_morphing, return_tensors='pt', padding='longest', truncation=True, max_length=512).to(device), max_length=(morph_length - 2) * max(batch['length']))
    morphed_sequences_decoded  = [[src] + x.split(" [SSEP] ") + [target] for (src, x, target) in zip(batch['premise'], morph_tokenizer.batch_decode(morphed_sequences, skip_special_tokens=True), batch['hypothesis'])]

    return morphed_sequences_decoded

@torch.inference_mode()
def nli_preds_over_morphs(morphs, nli_model, nli_tokenizer, nli_id2label, device):
    morphed_lengths = [len(x) for x in morphs]

    nli_model_input = []
    for x in morphs:
        for i in range(len(x) - 1):
            nli_model_input.append((x[i], x[i+1]))


    tokenized = nli_tokenizer(nli_model_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    output    = nli_model(**tokenized).logits.argmax(dim=1).split([x-1 for x in morphed_lengths])
    output    = [x.detach().detach().cpu().tolist() for x in output]
    output    = [[nli_id2label[y].lower() for y in x] for x in output]

    return output

def constitency_filter(output, gold_label, morph_length):
    if len(output) != morph_length - 1:
        return False
        
    count = Counter(output)
    if gold_label == 'entailment':
        if 'entailment' in count    and count['entailment'] == len(output):
            return True
    elif gold_label == 'neutral':
        if 'neutral' in count       and next((x for x in output if x != 'entailment')) == 'neutral':
            return True
    elif gold_label == 'contradiction':
        if 'contradiction' in count and next((x for x in output if x != 'entailment')) == 'contradiction':
            return True

    return False


if __name__ == "__main__":
    import tqdm
    import torch
    import datasets
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
    from sklearn.metrics import accuracy_score
    from collections import Counter
    import wandb
    device = torch.device('cuda:0')
    nli_model = AutoModelForSequenceClassification.from_pretrained("symanto/mpnet-base-snli-mnli").to(device)
    nli_tokenizer = AutoTokenizer.from_pretrained("symanto/mpnet-base-snli-mnli")

    nli_id2label = nli_model.config.id2label


    path = '/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/output_230414_1/checkpoint-50000'
    morph_tokenizer = AutoTokenizer.from_pretrained(path)
    morph_model     = T5ForConditionalGeneration.from_pretrained(path).to(device)

    sick_data   = datasets.load_dataset('sick')
    sick_labels = sick_data['train'].features['label'].names
    sick_data_val = sick_data['validation'].map(lambda x: {'p': x['sentence_A'], 'h': x['sentence_B'], 'l': sick_labels[x['label']].lower()}, load_from_cache_file=False).remove_columns(['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset']).rename_columns({'p': 'premise', 'h': 'hypothesis', 'l': 'label'})


    result_morph   = get_morph_nli_predictions(torch.utils.data.DataLoader(sick_data_val, batch_size=20), nli_model, nli_tokenizer, morph_model, morph_tokenizer, 3, nli_id2label, device)
    result_vanilla = get_vanilla_nli_predictions(torch.utils.data.DataLoader(sick_data_val, batch_size=20), nli_model, nli_tokenizer, nli_id2label, device)

    print(accuracy_score(sick_data_val['label'], result_morph))
    print(accuracy_score(sick_data_val['label'], result_vanilla))

    # 
    dl = torch.utils.data.DataLoader(sick_data_val, batch_size=2)
    batch = next(iter(dl))
    generated_morphings = generate_morphs(batch, morph_model, morph_tokenizer, morph_length=3)

    # We obtain the NLI prediction over the morphs previously generated by using an NLI model
    nli_pred_on_morphs = nli_preds_over_morphs(generated_morphings, nli_model, nli_tokenizer, nli_id2label)

    # We filter for constitency
    nli_data_from_morphs = [generated_morphings[i] for (i, x) in enumerate(nli_pred_on_morphs) if constitency_filter(x, batch['label'][i])]