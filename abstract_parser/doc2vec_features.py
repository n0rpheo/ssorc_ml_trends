import os
import pandas as pd
import pickle
import numpy as np

from statistics import mode
from statistics import StatisticsError

from spacy.vocab import Vocab
from spacy.tokens import Doc

from nltk.util import ngrams

from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense

from abstract_parser.learn_rules import load_rules

from utils.LoopTimer import LoopTimer
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


def doc_to_target_vector(input_doc, input_rules, trigger_words_=None):
    target_vector_ = list()
    pos_lists = list()
    lemma_lists = list()
    for sentence in input_doc.sents:
        # Collect All possible Categories
        cat_assignment = list()
        for category in input_rules:
            for rule in input_rules[category]:
                trigger_word = rule[0]
                dependency = rule[1]

                for token in sentence:
                    if token.text == trigger_word:
                        for child in token.children:
                            if child.dep_ == dependency:
                                cat_assignment.append(category)

        # Add most common category (or None if not available)
        try:
            target_vector_.append(mode(cat_assignment))
            pos_list = list()
            lemma_list = list()
            for token in sentence:
                if token_conditions(token, trigger_words_=trigger_words_):
                    lemma_list.append(token.lemma_)
                pos_list.append(token.tag_)
            lemma_lists.append(lemma_list)
            pos_lists.append(pos_list)
        except StatisticsError:
            target_vector_.append(None)
            lemma_lists.append(None)
            pos_lists.append(None)
    return target_vector_, lemma_lists, pos_lists


def doc_to_feature_vector(input_doc, target_vector_=None, trigger_words_=None):
    feature_vector_ = list()

    n_sents = sum([1 for i in input_doc.sents])

    for s_id, sentence in enumerate(input_doc.sents):

        if target_vector_ is None or target_vector_[s_id] is not None:
            location_ = s_id / n_sents
            word_vectors_ = [token.vector for token in sentence
                             if token_conditions(token, trigger_words_=trigger_words_)]

            if len(word_vectors_) > 0:
                doc_vector = sum(word_vectors_) / len(word_vectors_)
                features_ = np.append(doc_vector, [location_])
                features_ = doc_vector
                feature_vector_.append(features_)
            else:
                feature_vector_.append(None)
        else:
            feature_vector_.append(None)
    return feature_vector_


def docs_preprocessor(docs):

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc_ if not token.isdigit()] for doc_ in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc_ if len(token) > 3] for doc_ in docs]

    return docs


if __name__ == "__main__":
    """
    """
    nlp_model = "en_newmodel"

    allow_tw = False

    feature_info_name = f"doc2vec_{'tw' if allow_tw else 'notw'}_features_info.pickle"

    settings = dict()
    settings["feature_set"] = ["doc2vec_vector",
                               # "location"
                               ]
    data_size = 10000
    """
    """

    path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
    path_to_patterns = os.path.join(paths.to_root, "pattern_matching")
    path_to_learned_patterns = os.path.join(path_to_patterns, nlp_model)
    path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)

    if not os.path.isdir(path_to_rfl):
        print(f"Create Directory {path_to_rfl}")
        os.mkdir(path_to_rfl)

    rules_filename = "learned_rules_it4.json"
    rule_file_path = os.path.join(path_to_learned_patterns, rules_filename)

    rules = load_rules(rule_file_path)

    trigger_words = set([rule[0].lower() for category in rules for rule in rules[category]])

    print("Loading Vocab...")
    vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
    infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

    db_size = len(infoDF)

    predictions = dict()
    targets = dict()

    target_vector = list()
    feature_vector = list()

    docs_lemma = list()
    docs_pos = list()

    lc = LoopTimer(update_after=100, avg_length=5000, target=min(data_size, db_size))
    for idx, (abstract_id, df_row) in enumerate(infoDF.iterrows()):
        doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

        doc_target_vector, doc_lemma_list, doc_pos_list = doc_to_target_vector(doc, rules,
                                                                               trigger_words_=None if allow_tw else trigger_words)

        doc_feature_vector = doc_to_feature_vector(doc, target_vector_=doc_target_vector,
                                                   trigger_words_=None if allow_tw else trigger_words)

        for tv, fv, lv, pv in zip(doc_target_vector, doc_feature_vector, doc_lemma_list, doc_pos_list):
            if tv is not None and fv is not None:
                target_vector.append(tv)
                feature_vector.append(fv)

                docs_lemma.append(lv)
                docs_pos.append(pv)

        info_count = [(cat, target_vector.count(cat)) for cat in set(target_vector)]
        breaker = lc.update(f"Make Features - {info_count}")

        if breaker > data_size:
            break

    dense_corpus_sent2vec = np.array(feature_vector)
    target_vector = np.array(target_vector)

    print()
    print(dense_corpus_sent2vec.shape)
    print(target_vector.shape)

    print(len(docs_lemma))
    print(len(docs_pos))

    print('Make Dictionary')
    dictionary_lemma = Dictionary(docs_lemma)
    dictionary_pos = Dictionary(docs_pos)
    print('Number of unique pos: %d' % len(dictionary_pos))
    dictionary_lemma.filter_extremes(no_below=10, no_above=0.2, keep_tokens=trigger_words if allow_tw else None)
    print('Number of unique lemma: %d' % len(dictionary_lemma))

    lemma_bigrams = list()
    for d in docs_lemma:
        lemma_bigram = [f"{bigram[0]}_{bigram[1]}" for bigram in list(ngrams(d, 2))]
        lemma_bigrams.append(lemma_bigram)

    pos_bigrams = list()
    for d in docs_pos:
        pos_bigram = [f"{bigram[0]}_{bigram[1]}" for bigram in list(ngrams(d, 2))]
        pos_bigrams.append(pos_bigram)

    dictionary_lemma_bigram = Dictionary(lemma_bigrams)
    dictionary_lemma_bigram.filter_extremes(no_below=2, no_above=0.2)
    print('Number of unique lemma_bigram: %d' % len(dictionary_lemma_bigram))
    print(dictionary_lemma_bigram)

    dictionary_pos_bigram = Dictionary(pos_bigrams)
    print('Number of unique pos_bigram: %d' % len(dictionary_pos_bigram))

    corpus_lemma = [dictionary_lemma.doc2bow(ld) for ld in docs_lemma]
    corpus_pos = [dictionary_pos.doc2bow(ld) for ld in docs_pos]
    corpus_lemma_bigram = [dictionary_lemma_bigram.doc2bow(ld) for ld in lemma_bigrams]
    corpus_pos_bigram = [dictionary_pos_bigram.doc2bow(ld) for ld in pos_bigrams]

    dense_corpus_lemma = corpus2dense(corpus_lemma, num_terms=len(dictionary_lemma))
    dense_corpus_lemma_bigram = corpus2dense(corpus_lemma_bigram, num_terms=len(dictionary_lemma_bigram))
    dense_corpus_pos = corpus2dense(corpus_pos, num_terms=len(dictionary_pos))
    dense_corpus_pos_bigram = corpus2dense(corpus_pos_bigram, num_terms=len(dictionary_pos_bigram))

    corpus = np.transpose(np.concatenate((dense_corpus_lemma,
                                          dense_corpus_lemma_bigram,
                                          dense_corpus_pos,
                                          dense_corpus_pos_bigram)))

    print(corpus.shape)

    feature_dict = dict()

    feature_dict["features_svec"] = dense_corpus_sent2vec
    feature_dict["features_ngram"] = corpus
    feature_dict["dict_lemma"] = dictionary_lemma
    feature_dict["dict_lemma_bigram"] = dictionary_lemma_bigram
    feature_dict["dict_pos"] = dictionary_pos
    feature_dict["dict_pos_bigram"] = dictionary_pos_bigram
    feature_dict["targets"] = target_vector
    feature_dict["settings"] = settings

    with open(os.path.join(path_to_rfl, f"{feature_info_name}"), "wb") as handle:
        pickle.dump(feature_dict, handle)
