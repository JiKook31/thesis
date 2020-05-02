import os
import random
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.augmentation import transformation_function
from snorkel.augmentation import RandomPolicy
from snorkel.augmentation import MeanFieldPolicy
from snorkel.augmentation import PandasTFApplier

# Turn off TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For reproducibility
seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(0)
random.seed(0)

DISPLAY_ALL_TEXT = False
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

df_train = open("data/train.txt")
df_test = open("data/test.txt")

Y_train = df_train["label"].values
Y_test = df_test["label"].values

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)
replacement_names = ["API", "Plat", "PL", "Stan", "Fram"]

# Replace a random named entity with a different entity of the same type.
@transformation_function(pre=[spacy])
def change_named_entity(x):
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == replacement_names]
    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.text = x.text.replace(name_to_replace, replacement_name)
        return x


def get_synonym(word, pos=None):
    synsets = wn.synsets(word, pos=pos)
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            return words[0].replace("_", " ")


def replace_token(spacy_doc, idx, replacement):
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])


@transformation_function(pre=[spacy])
def replace_with_synonym(x):
    idxs = [i for i, token in enumerate(x.doc)]
    if idxs:
        idx = np.random.choice(idxs)
        synonym = get_synonym(x.doc[idx].text)
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


tfs = [
    change_named_entity,
    replace_with_synonym,
]

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True
)

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
    p=[0.05, 0.05, 0.3, 0.3, 0.3],
)



tf_applier = PandasTFApplier(tfs, mean_field_policy)
df_train_augmented = tf_applier.apply(df_train)
Y_train_augmented = df_train_augmented["label"].values

print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")
