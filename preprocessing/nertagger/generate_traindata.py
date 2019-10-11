import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
import pandas as pd

from utils.LoopTimer import LoopTimer
from info import paths


nlp_model = "en_core_web_lg"
path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
pandas_path = os.path.join(paths.to_root, "pandas")
path_to_ner = os.path.join(paths.to_root, "NER")

threshold = 40

print("Loading NLP Model and Vocab...")
#nlp = spacy.load(os.path.join(paths.to_root, "models", nlp_model))
nlp = spacy.load(nlp_model)
vocab = nlp.vocab.from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
matcher = Matcher(vocab)

# This file is required and filled with basic algorithm names (including acronyms)
basic_ml_algos_file = "ml_algos_withacronyms.txt"

mla = set()
with open(os.path.join(path_to_ner, basic_ml_algos_file), "r") as handle:
    for line in handle:
        mla.add(line.replace("\n", ""))

for ml_algo in mla:
    ml_doc = nlp(ml_algo)
    pattern = [{"LOWER": token.lower_} for token in ml_doc]
    pattern_name = "".join([entity["LOWER"] for entity in pattern]).lower()
    matcher.add(pattern_name, None, pattern)

infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
targ = len(infoDF)
ent_counter = dict()

#forbidden_dep = ['csubj', 'nummod', 'cc', 'advmod', 'preconj', 'attr', 'det']
#forbidden_pos = ['VERB', 'ADP']

forbidden_dep = ['det', 'predet', 'nummod', 'cc', 'appos', 'punct', 'conj']
forbidden_pos = ['ADP', 'VERB', 'X', 'ADV']

forbidden_substrings = ['state-of-the-art', ',', '(', ')',
                        "approaches",
                        "approach",
                        "algorithm",
                        "algorithms",
                        "based",
                        "function",
                        "functions",
                        "other",
                        "large",
                        "larger",
                        "twitter",
                        "such"]

collect_ml = set()
sentence_id = 0
train_list = list()
s_sid_list = list()

lt = LoopTimer(update_after=200, avg_length=2000, target=targ)

# Iterating over all abstracts
for abstract_id, row in infoDF.iterrows():
    ori_doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))


    # Training set is built on sentences, so iterate over sentences
    for sent in ori_doc.sents:
        sentence = sent.as_doc()

        # get ML matches
        matches = matcher(sentence)

        ent_list = list()

        # Iterate over Matches
        for match in matches:
            start = match[1]
            end = match[2]

            #  Add all tokens to this set, that should be processed
            #  Extend this set with feasible tokens and add all processed tokens to visited
            #  and all token-ids to algo_set

            to_process = set()
            visited = set()
            algo_set = set()

            for i in range(start, end):
                to_process.add(sentence[i])

            while len(to_process) > 0:
                token = to_process.pop()
                visited.add(token)
                algo_set.add(token.i)

                if token.dep_ == "compound":
                    if (token.head not in visited
                            and token.head not in to_process):
                        to_process.add(token.head)
                children = token.children
                for child in children:
                    if (child.dep_ not in forbidden_dep
                            and child.pos_ not in forbidden_pos
                            and child not in visited
                            and child not in to_process):
                        to_process.add(child)
            algo_list = list(algo_set)
            algo_list.sort()

            start_id = min(algo_list)
            end_id = max(algo_list)

            entity_name = sentence[start_id:end_id+1].text.lower()
            entity_text_tokens = [etoken.orth_.lower() for etoken in sentence[start_id:end_id + 1]]
            entity_pos_tokens = [etoken.pos_ for etoken in sentence[start_id:end_id + 1]]

            #  Check if the resulting entitiy is feasible and then add it to the training set

            if (all([(forbidden_string not in entity_text_tokens) for forbidden_string in forbidden_substrings]) and
                    all([(forbidden_pos_t not in entity_pos_tokens) for forbidden_pos_t in forbidden_pos]) and
                    ("markov model" in entity_name) ** ("model" in entity_name or "models" in entity_name)):
                start_idx = sentence[start_id].idx
                end_idx = sentence[end_id].idx + len(sentence[end_id])

                entity = (start_idx, end_idx, entity_name)
                ent_list.append(entity)
        if len(ent_list) > 0:
            word_list = list()
            tag_list = list()

            entity_ids = list()
            for entity in ent_list:
                entity_name = entity[2]
                if entity_name in ent_counter:
                    ent_counter[entity_name]["sent_list"].append(sentence_id)
                    ent_counter[entity_name]["count"] += 1
                else:
                    ent_counter[entity_name] = dict()
                    ent_counter[entity_name]["sent_list"] = [sentence_id]
                    ent_counter[entity_name]["count"] = 1

            entities = {"entities": [(e[0], e[1], "MLALGO") for e in ent_list]}
            train_list.append((sentence.text, entities))
            s_sid_list.append(sentence_id)
            sentence_id += 1

    breaker = lt.update(f"Make TD - {len(train_list)}")

#  Iterate over all possible training examples and only add those that have a minimal number of occurrences

for entity in ent_counter:
    count = ent_counter[entity]["count"]
    eslist = ent_counter[entity]["sent_list"]

    if count < threshold:
        for sid in eslist:
            if sid in s_sid_list:
                s_sid_list.remove(sid)
    else:
        print(f"{entity}: {count}")


TRAIN_DATA = list()
for sid in s_sid_list:
    TRAIN_DATA.append(train_list[sid])

with open(os.path.join(path_to_ner, "mlalgo_spacy_traindata.pickle"), "wb") as handle:
    pickle.dump(TRAIN_DATA, handle)
