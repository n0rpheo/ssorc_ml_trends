from __future__ import absolute_import, division, print_function
import os
import pickle

import pandas as pd
import numpy as np

from gensim.corpora.dictionary import Dictionary

import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc

from MLRes.occ.corpus_and_dictionary import replace_cluster_in_doc, doc_2_token, load_cluster_dic

from topic_modeling.lda_features import lda_feature_vector

from utils.LoopTimer import LoopTimer

from info import paths


nlp_model = "en_newmodel"
v_num_topics = 500

# Input Mod1

aid_list_fn = "aid_list.pickle"
rf_list_fn = "rf_list.pickle"

tm_info_fn = f"lemma_vec_occ2_{v_num_topics}.info"

# Output
per_year_fn = f"pop_per_year_p2_{v_num_topics}.pickle"
per_sentence_fn = f"pop_per_sentence_p2_{v_num_topics}.pickle"
per_abstract_fn = f"pop_per_abstract_p2_{v_num_topics}.pickle"

# All Paths
path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)
path_to_pandas = os.path.join(paths.to_root, "pandas", nlp_model)
path_to_mlgenome = os.path.join(paths.to_root, "mlgenome", nlp_model)
path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)


print("Loading Vocab...")
nlp = spacy.load(os.path.join(paths.to_root, "models", nlp_model))
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

with open(os.path.join(path_to_popularities, aid_list_fn), 'rb') as handle:
    aid_list = pickle.load(handle)

print(len(aid_list))


with open(os.path.join(path_to_popularities, rf_list_fn), 'rb') as handle:
    rf_list = pickle.load(handle)

with open(os.path.join(path_to_tm, tm_info_fn), "rb") as handle:
    tm_info = pickle.load(handle)

for key in tm_info:
    print(f"{key}: {tm_info[key]}")

with open(tm_info["model_path"], "rb") as model_file:
    lda_model = pickle.load(model_file)

mentions, sorted_mentions, replace_dic = load_cluster_dic(path=tm_info['occ_matrix_path'])
dictionary = Dictionary.load(tm_info['dic_path'])

years = sorted(np.unique(infoDF['year'].values))

lda_per_year = dict()
# lda_per_sentence = dict()
lda_per_abstract = dict()

for year in years:
    if year == 2018:
        continue
    aid_of_year = set(infoDF.loc[infoDF['year'] == year].index.tolist()) & set(aid_list)

    if len(aid_of_year) > 0:
        lda_per_year[year] = dict()
        # lda_per_sentence[year] = dict()
        lda_per_abstract[year] = dict()
        print()
    else:
        continue

    segments_per_year = dict()
    segments_per_year['num_abstracts'] = len(aid_of_year)
    segments_per_year['rf'] = dict()

    # segments_per_sentence = dict()
    # segments_per_sentence['num_abstracts'] = len(aid_of_year)
    # segments_per_sentence['rf'] = dict()

    segments_per_abstract = dict()
    segments_per_abstract['num_abstracts'] = len(aid_of_year)
    segments_per_abstract['rf'] = dict()

    lc = LoopTimer(update_after=100, avg_length=200, target=len(aid_of_year))
    for num_abstracts, abstract_id in enumerate(aid_of_year):

        rf_index = aid_list.index(abstract_id)

        rf_pred = rf_list[rf_index]

        doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

        token_list = list()
        for sentence in doc.sents:
            sent_as_doc = sentence.as_doc()
            sent_as_doc = replace_cluster_in_doc(sent_as_doc, replace_dic, sorted_mentions, nlp)
            token_list.append(doc_2_token(sent_as_doc, split_sentences=False))

        rfp_abstract_id = dict()

        for rfp, tl in zip(rf_pred, token_list):
            if rfp not in segments_per_year['rf']:
                segments_per_year['rf'][rfp] = list()
                # segments_per_sentence['rf'][rfp] = list()
                segments_per_abstract['rf'][rfp] = list()
            if rfp not in rfp_abstract_id:
                rfp_abstract_id[rfp] = set()

            if abstract_id not in rfp_abstract_id[rfp]:
                rfp_abstract_id[rfp].add(abstract_id)
                segments_per_abstract['rf'][rfp].append(list())
            if len(tl) > 0:
                segments_per_abstract['rf'][rfp][-1].extend(tl)
                segments_per_year['rf'][rfp].extend(tl)
                # segments_per_sentence['rf'][rfp].append(tl)
        lc.update(f"Collect TM Data - {year}")

    lc = LoopTimer(update_after=1, avg_length=1, target=len(segments_per_year['rf']))
    for rfp in segments_per_year['rf']:
        lda_vec = lda_feature_vector([segments_per_year['rf'][rfp]], dictionary)
        lda_per_year[year][rfp] = lda_model.transform(lda_vec)
        lc.update(f"Transform per year - {year}")

    # lc = LoopTimer(update_after=1, avg_length=1, target=len(segments_per_year['rf']))
    # for rfp in segments_per_sentence['rf']:
    #     lda_vec = lda_feature_vector(segments_per_sentence['rf'][rfp], dictionary)
    #     lda_per_sentence[year][rfp] = lda_model.transform(lda_vec)
    #     lc.update(f"Transform per sentence - {year}")

    lc = LoopTimer(update_after=1, avg_length=1, target=len(segments_per_abstract['rf']))
    for rfp in segments_per_abstract['rf']:
        lda_vec = lda_feature_vector(segments_per_abstract['rf'][rfp], dictionary)
        lda_per_abstract[year][rfp] = lda_model.transform(lda_vec)
        lc.update(f"Transform per abstract - {year}")

if not os.path.isdir(path_to_popularities):
    print(f"Create Directory {path_to_popularities}")
    os.mkdir(path_to_popularities)

with open(os.path.join(path_to_popularities, per_year_fn), 'wb') as handle:
    pickle.dump(lda_per_year, handle)

# with open(os.path.join(path_to_popularities, per_sentence_fn), 'wb') as handle:
#     pickle.dump(lda_per_sentence, handle)

with open(os.path.join(path_to_popularities, per_abstract_fn), 'wb') as handle:
    pickle.dump(lda_per_abstract, handle)
