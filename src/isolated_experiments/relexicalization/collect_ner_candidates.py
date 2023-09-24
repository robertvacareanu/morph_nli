"""
In order to relexicalize the Text Morphing data, we need to have entities
In order to do this, we take oscar dataset for english, shuffle it, then take and annotate 10_000 examples

This resulted in the following sizes for each entity
```
CARDINAL 10017
DATE 20485
EVENT 1540
FAC 4300
GPE 13399
LANGUAGE 66
LAW 1176
LOC 3601
MONEY 3571
NORP 3770
ORDINAL 265
ORG 60079
PERCENT 1753
PERSON 48860
PRODUCT 3554
QUANTITY 3672
TIME 3373
WORK_OF_ART 6125
```
"""

import spacy
import datasets
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

NER = spacy.load("en_core_web_sm", disable=['parser'])


language = 'en'
size = 10_000
data = []
for line in tqdm.tqdm(datasets.load_dataset("oscar", f"unshuffled_deduplicated_{language}", split="train", streaming=True).shuffle(seed=1, buffer_size=50_000).take(size), total=size):
    data.append(line['text'])

print("Loaded")

data_annotated = list(NER.pipe(data, batch_size=25, n_process=8))

print("Annotated")

data_annotated_dict = defaultdict(set)
for x in data_annotated:
    for y in x.ents:
        data_annotated_dict[y.label_].add(y.text)

import pickle
with open('random_entities.pkl', 'wb+') as fout:
    pickle.dump(data_annotated_dict, fout)
