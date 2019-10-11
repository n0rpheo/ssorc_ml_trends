import os
import json
import spacy
import numpy as np
import pickle
from gensim.matutils import corpus2dense
from nltk.util import ngrams

from info import paths


def token_conditions(token_, trigger_words_=None):
    if token_.ent_iob == 3 or token_.ent_iob == 1:
        return True
    if token_.is_punct:
        return False
    if token_.is_stop:
        return False
    if trigger_words_ is not None and token_.orth_.lower() in trigger_words_:
        return False

    return True


nlp_model = "en_newmodel"

path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
path_to_patterns = os.path.join(paths.to_root, "pattern_matching")
path_to_learned_patterns = os.path.join(path_to_patterns, nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)
path_to_dt = "/home/norpheo/Documents/thesis/rf_labeling.json"              # File Required - Handlabeled with DataTurks
nlp_path = os.path.join(paths.to_root, "models", nlp_model)

print("Loading NLP...")
nlp = spacy.load(nlp_path)

rules_filename = "learned_rules_it4.json"
rule_file_path = os.path.join(path_to_learned_patterns, rules_filename)

rules = dict()
with open(rule_file_path) as rule_file:
    for line in rule_file:
        data = json.loads(line)

        category = data['category']
        trigger_word = data['trigger_word']
        dependency = data['dependency']

        rule = (trigger_word, dependency)

        if category not in rules:
            rules[category] = set()

            rules[category].add(rule)

trigger_words = set([rule[0].lower() for category in rules for rule in rules[category]])

allow_tw = False
feature_info_name = f"doc2vec_{'tw' if allow_tw else 'notw'}_features_info.pickle"

with open(os.path.join(path_to_rfl, feature_info_name), "rb") as feature_file:
    feature_dict = pickle.load(feature_file)

with open(path_to_dt, "r") as handle:
    feature_vector_sv = list()
    feature_vector_ngram = list()
    target_vector = list()

    docs_lemma = list()
    docs_pos = list()

    for line in handle:
        line_data = json.loads(line)
        sentence_string = line_data['content']
        rf_label = line_data['annotation']['labels'][0]

        target_vector.append(rf_label)

        doc = nlp(sentence_string)

        docs_lemma.append([token.lemma_ for token in doc])
        docs_pos.append((token.tag_ for token in doc))

        location_ = 0
        word_vectors_ = [token.vector for token in doc if token_conditions(token,
                                                                           trigger_words_=None if allow_tw else trigger_words)]

        doc_vector = sum(word_vectors_) / len(word_vectors_)
        feature_vector_sv.append(doc_vector)

    dense_corpus_sent2vec = np.array(feature_vector_sv)
    target_vector = np.array(target_vector)

    print()
    print(dense_corpus_sent2vec.shape)
    print(target_vector.shape)
    print(len(docs_lemma))
    print(len(docs_pos))

    """
        Ngram Features
    """

    print('Make Dictionary')
    dictionary_lemma = feature_dict["dict_lemma"]
    dictionary_pos = feature_dict["dict_pos"]
    print('Number of unique pos: %d' % len(dictionary_pos))
    print('Number of unique lemma: %d' % len(dictionary_lemma))

    lemma_bigrams = list()
    for d in docs_lemma:
        lemma_bigram = [f"{bigram[0]}_{bigram[1]}" for bigram in list(ngrams(d, 2))]
        lemma_bigrams.append(lemma_bigram)

    pos_bigrams = list()
    for d in docs_pos:
        pos_bigram = [f"{bigram[0]}_{bigram[1]}" for bigram in list(ngrams(d, 2))]
        pos_bigrams.append(pos_bigram)

    dictionary_lemma_bigram = feature_dict['dict_lemma_bigram']
    print('Number of unique lemma_bigram: %d' % len(dictionary_lemma_bigram))

    dictionary_pos_bigram = feature_dict['dict_pos_bigram']
    print('Number of unique pos_bigram: %d' % len(dictionary_pos_bigram))

    corpus_lemma = [dictionary_lemma.doc2bow(ld) for ld in docs_lemma]
    corpus_pos = [dictionary_pos.doc2bow(ld) for ld in docs_pos]
    corpus_lemma_bigram = [dictionary_lemma_bigram.doc2bow(ld) for ld in lemma_bigrams]
    corpus_pos_bigram = [dictionary_pos_bigram.doc2bow(ld) for ld in pos_bigrams]

    dense_corpus_lemma = corpus2dense(corpus_lemma, num_terms=len(dictionary_lemma))
    dense_corpus_lemma_bigram = corpus2dense(corpus_lemma_bigram, num_terms=len(dictionary_lemma_bigram))
    dense_corpus_pos = corpus2dense(corpus_pos, num_terms=len(dictionary_pos))
    dense_corpus_pos_bigram = corpus2dense(corpus_pos_bigram, num_terms=len(dictionary_pos_bigram))

    corpus = np.transpose(
        np.concatenate((dense_corpus_lemma, dense_corpus_lemma_bigram, dense_corpus_pos, dense_corpus_pos_bigram)))

    print(corpus.shape)

    gold_label_name = f"gold_labels_{'tw' if allow_tw else 'notw'}.pickle"

    gold_label_dict = dict()
    gold_label_dict["features_svec"] = dense_corpus_sent2vec
    gold_label_dict["features_ngram"] = corpus
    gold_label_dict["targets"] = target_vector

    with open(os.path.join(path_to_rfl, gold_label_name), "wb") as handle2:
        pickle.dump(gold_label_dict, handle2)
