import os
import pandas as pd
import spacy
import pickle
import numpy as np
from spacy.tokens import Doc
from gensim.corpora.dictionary import Dictionary

from utils.LoopTimer import LoopTimer
from info import paths


def token_conditions(token_):
    if token_.ent_iob == 3 or token_.ent_iob == 1:
        return True
    if token_.is_punct:
        return False
    if token_.is_stop:
        return False
    if len(token_.orth_) < 3:
        return False

    return True


def replace_cluster_in_doc(input_doc, replace_dic_, sorted_mentions_, nlp_):
    ds = input_doc.text

    change_b = False
    offset = 0

    for entity in input_doc.ents:

        start_idx = input_doc[entity.start].idx
        end_idx = input_doc[entity.end - 1].idx + len(input_doc[entity.end - 1])
        es = entity.text.lower()
        replacement = entity.text

        for mention in sorted_mentions_:
            ms = mention["string"]
            if ms == es:
                if es in replace_dic_:
                    replacement = " ".join(f"aimlcluster{cluster_id}" for cluster_id in replace_dic_[es])
                    replacement = " " + replacement + " "
                break

        ds = ds[:start_idx + offset] + replacement + ds[end_idx + offset:]
        offset += len(replacement) - len(entity.text)
        change_b = True

    if change_b:
        input_doc = nlp_(ds)

    return input_doc


def load_cluster_dic(path):
    with open(path, "rb") as handle_:
        occ_res = pickle.load(handle_)

    mentions_ = occ_res["mentions"]
    cluster_matrix_ = occ_res["cluster_matrix"]

    replace_dic_ = dict()

    sorted_mentions_ = sorted(mentions_, key=lambda s: len(s['string']), reverse=True)

    for mention_id_, mention_ in enumerate(mentions_):
        cluster_ids_ = np.where(cluster_matrix_[mention_id_, :] == 1)[0]
        replace_dic_[mention_["string"]] = cluster_ids_

    return mentions_, sorted_mentions_, replace_dic_


def doc_2_token(input_doc, split_sentences=False):
    if split_sentences:
        return [[token.lemma_ for token in sentence if token_conditions(token)] for sentence in input_doc.sents]
    else:
        return [token.lemma_ for token in input_doc if token_conditions(token)]


if __name__ == '__main__':
    nlp_model = "en_newmodel"
    occ_file_name = "occ_p2_best.pickle"

    corpus_file_name = "corpus_p2.pickle"  # OUTPUT
    dictionary_file_name = "lemma_dict_p2.gendic"  # OUTPUT

    path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
    path_to_pandas = os.path.join(paths.to_root, "pandas", nlp_model)
    path_to_mlgenome = os.path.join(paths.to_root, "mlgenome", nlp_model)

    if not os.path.isdir(path_to_pandas):
        print(f"Create Directory {path_to_pandas}")
        os.mkdir(path_to_pandas)

    mentions, sorted_mentions, replace_dic = load_cluster_dic(path=os.path.join(path_to_mlgenome, occ_file_name))

    nlp = spacy.load(os.path.join(paths.to_root, "models", nlp_model))
    vocab = nlp.vocab.from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
    infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

    lemma_s_list = list()
    lemma_d_list = list()
    abstract_id_list = list()

    target = len(infoDF)
    lt = LoopTimer(update_after=10, avg_length=1000, target=target)
    for abstract_id, row in infoDF.iterrows():
        doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

        doc = replace_cluster_in_doc(doc, replace_dic, sorted_mentions, nlp)

        lemma_s_list.append(doc_2_token(doc, split_sentences=True))
        lemma_d_list.append(doc_2_token(doc, split_sentences=False))
        abstract_id_list.append(abstract_id)

        breaker = lt.update(f"Create Pandas - {len(lemma_d_list)}")

    dictionary = Dictionary(lemma_d_list)
    id_d_list = [dictionary.doc2idx(document) for document in lemma_d_list]
    id_s_list = [[dictionary.doc2idx(sentence) for sentence in document] for document in lemma_s_list]

    corpus = {"abstract_id": abstract_id_list,
              "lemma_sentence": lemma_s_list,
              "lemma_document": lemma_d_list,
              "lemma_id_sentence": id_s_list,
              "lemma_id_document": id_d_list}

    with open(os.path.join(path_to_pandas, corpus_file_name), "wb") as handle:
        pickle.dump(corpus, handle)

    dictionary.save(os.path.join(path_to_pandas, dictionary_file_name))
