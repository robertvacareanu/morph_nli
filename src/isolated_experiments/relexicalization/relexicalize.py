import datasets
import random
import json
import pickle
import tqdm
from datasets import Dataset, DatasetDict
from typing import Dict, List

random.seed(1)

with open('/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/random_entities.pkl', 'rb') as fin:
    lexicon = pickle.load(fin)
    lexicon = list(lexicon.items())
    lexicon = [(f'<{x[0]}>', sorted(list(x[1]))) for x in lexicon]
    lexicon = dict(sorted(lexicon, key=lambda x: x[0]))

tags = list(lexicon.keys())

def relexicalize_line(line: str, lexicon: Dict[str, List[str]], tags):
    split  = line.split('\t')
    source = split[0]
    target = split[-1]

    max_ne_count = max([sent.lower().count(tag.lower()) for sent in split for tag in tags])
    if max_ne_count == 1:
        # Relexicalize
        # We already know that there is at most of every tag. We do this to prevent 
        # tricky situations where their relative order is reversed
        # John works for Google but wants to move to Microsoft -> John wants to move to Microsoft but works for Google
        # (Those would be represented as <PERSON> works for <ORG> but wants to move to <ORG> -> <PERSON> wants to move to <ORG> but works for <ORG>)

        # We also sort, just to avoid unreproducibility introduced by a call to set
        tags_in       = sorted(list(set([tag for sent in split for tag in tags if tag in sent])))
        tags_to_words = {tag:random.choice(lexicon[tag]) for tag in tags_in}

        split_replaced = [x for x in split]
        for tag in tags_in:
            should_replace_all = random.random() > 0.5
            if should_replace_all:
                split_replaced = [x.replace(tag, tags_to_words[tag]) for x in split_replaced]
            else:
                index = random.choice(range(len(split)))
                split_replaced = [x.replace(tag, tags_to_words[tag]) if i < index else x.replace(tag, random.choice(lexicon[tag])) for (i, x) in enumerate(split_replaced)]
        
        source = split_replaced[0]
        target = split_replaced[-1]

        return split_replaced
    elif max_ne_count == 0:
        return split
    else:
        return split

def relexicalize(line: str):
    result = relexicalize_line(line, lexicon, tags)
    return result

train = []
with open('/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/data/train.txt') as fin:
    for line in tqdm.tqdm(fin):
        train.append(relexicalize(line))

dev = []
with open('/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/data/dev.txt') as fin:
    for line in fin:
        dev.append(relexicalize(line))

test_1 = []
with open('/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/data/test_1.txt') as fin:
    for line in fin:
        test_1.append(relexicalize(line))

test_2 = []
with open('/home/rvacareanu/projects_6_2301/alignment/natlog/text_morph/data/test_2.txt') as fin:
    for line in fin:
        test_2.append(relexicalize(line))


# dataset = DatasetDict(
#     {
#         'train'     : Dataset.from_list(train),
#         'validation': Dataset.from_list(dev),
#         'test1'     : Dataset.from_list(test1),
#         'test2'     : Dataset.from_list(test2),
#     }
# ).filter(lambda x: x['lexicalized']).filter(lambda x: not any(skip in x['output'] for skip in tags)).remove_columns(['entities', 'lexicalized'])

# print(dataset)
# dataset.save_to_disk('')

for (name, dataset) in [('train', train), ('dev', dev), ('test_1', test_2), ('test_2', test_2)]:
    with open(f'/data/nlp/morphing/230923/data/text_morph_relexicalized/{name}.txt', 'w+') as fout:
        for line in dataset:
            _ = fout.write(json.dumps('\t'.join(line)))
            _ = fout.write('\n')
