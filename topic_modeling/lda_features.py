import os
import math
import pickle

import numpy as np
import scipy.sparse

import seaborn as sns

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

from info import paths


def tokens_to_tfidf(input_tokens, dictionary_):
    bow = dictionary_.doc2bow(input_tokens)
    if len(bow) == 0:
        return []
    tf = [t[1] for t in bow]
    max_t = max(tf)
    n_doc = dictionary_.num_docs

    return [(t[0], (t[1] / max_t) * math.log(n_doc / dictionary_.dfs[t[0]])) for t in bow]


def tokens_to_bow(input_tokens, dictionary_):
    bow = dictionary_.doc2bow(input_tokens)
    if len(bow) == 0:
        return []
    return bow


def lda_feature_vector(in_list, dictionary_):
    row = []
    col = []
    data = []

    for idx, tokens in enumerate(in_list):
        # tfidf_vec = tokens_to_tfidf(tokens, dictionary_)
        bow_vec = tokens_to_bow(tokens, dictionary_)
        word_vec = bow_vec
        for entry in word_vec:
            row.append(idx)
            col.append(entry[0])
            data.append(entry[1])

    m = idx + 1
    n = len(dictionary_)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    feature_vector_ = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))

    return feature_vector_


def docs_preprocessor(docs):

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    return docs


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(f"Train {num_topics}")
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

        x = range(start, num_topics+step, step)
        coherence_pairs = (x, coherence_values)
        with open(os.path.join("/home/norpheo/Documents/thesis", "coherence_pair_umass.pickle"), "wb") as handle:
            pickle.dump(coherence_pairs, handle)

    return model_list, coherence_values


if __name__ == '__main__':
    """
        INFOS
        #################################
    """

    nlp_model = "en_newmodel"
    dic_name = "lemma_dict_p2.gendic"     # occ.corpus_and_dictionary.py
    corpus_name = "corpus_p2.pickle"      # occ.corpus_and_dictionary.py
    occ_file_name = "occ_p2_best.pickle"  # occ.occ.py

    feature_file_name = f"features_lemma_vec_p2.npz"  # output
    info_name = f"lemma_vec_occ2.info"  # output

    """
        #################################
    """

    path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
    path_to_pandas = os.path.join(paths.to_root, "pandas", nlp_model)
    path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
    path_to_features = os.path.join(path_to_tm, "features")
    path_to_mlgenome = os.path.join(paths.to_root, "mlgenome", nlp_model)

    if not os.path.isdir(path_to_tm):
        print(f"Create Directory {path_to_tm}")
        os.mkdir(path_to_tm)
    if not os.path.isdir(path_to_features):
        print(f"Create Directory {path_to_features}")
        os.mkdir(path_to_features)

    dictionary = Dictionary.load(os.path.join(path_to_pandas, dic_name))
    with open(os.path.join(path_to_pandas, corpus_name), "rb") as handle:
        corpus = pickle.load(handle)

    lemma_d_list = corpus['lemma_document']  # [:10000]

    docs = docs_preprocessor(lemma_d_list)

    occ_tokens = set()
    for doc in docs:
        for token in doc:
            if 'aiml' in token:
                occ_tokens.add(token)

    occ_tokens = list(occ_tokens)
    print('Make Dictionary')
    dictionary = Dictionary(docs)
    print('Number of unique tokens: %d' % len(dictionary))
    dictionary.filter_extremes(no_below=10, no_above=0.2, keep_tokens=occ_tokens)
    print('Number of unique tokens: %d' % len(dictionary))

    feature_vector = lda_feature_vector(docs, dictionary)

    scipy.sparse.save_npz(os.path.join(path_to_features, feature_file_name), feature_vector)
    info = dict()
    dic_name = "lemma_dict_p2_new.gendic"
    dictionary.save(os.path.join(path_to_pandas, dic_name))
    info['dic_path'] = os.path.join(path_to_pandas, dic_name)
    info['feature_path'] = os.path.join(path_to_features, feature_file_name)
    info['occ_matrix_path'] = os.path.join(path_to_mlgenome, occ_file_name)

    with open(os.path.join(path_to_tm, info_name), "wb") as handle:
        pickle.dump(info, handle)
