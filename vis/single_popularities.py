import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gensim

from info import paths

sns.set()

nlp_model = "en_newmodel"

# RF Labels

rf_labelnames_dict_fn = f"ann_model_labelnames.dict"

# Topic Cluster
cluster_type = "p2"
v_num_topics = 500

if cluster_type == "p1":
    """
        Cluster P1
    """

    per_year_fn = f"pop_per_year_p2_{v_num_topics}.pickle"
    per_sentence_fn = f"pop_per_sentence_p2_{v_num_topics}.pickle"
    per_abstract_fn = f"pop_per_abstract_p2_{v_num_topics}.pickle"
    tm_info_fn = f"lemma_vec_occ2_{v_num_topics}.info"

    topic_cluster_names = [("aimlcluster368", "Neural Network"),
                           ("aimlcluster1029", "Stochastic Gradient Descent"),
                           ("aimlcluster1926", "Decision Tree Learning"),
                           ("aimlcluster2350", "linear regression"),
                           ("aimlcluster2687", "KNN"),
                           ("aimlcluster2923", "Naive Bayes"),
                           ("aimlcluster3209", "SVM")
                           ]
else:
    """
        Cluster P2
    """

    per_year_fn = f"pop_per_year_p2_{v_num_topics}.pickle"
    per_sentence_fn = f"pop_per_sentence_p2_{v_num_topics}.pickle"
    per_abstract_fn = f"pop_per_abstract_p2_{v_num_topics}.pickle"
    tm_info_fn = f"lemma_vec_occ2_{v_num_topics}.info"

    topic_cluster_names = [("aimlcluster105", "SVM"),
                           ("aimlcluster93", "PCA"),
                           ("aimlcluster108", "Decision Trees"),
                           ("aimlcluster38", "Naive Bayes"),
                           ("aimlcluster122", "Neural Network")
                           ]

# Year Range
year_start = 1980
year_end = 2017

# All Paths
path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)

path_to_fig_save = "/home/norpheo/Documents/thesis/Results/fig/popularities/new_model"

with open(os.path.join(path_to_tm, tm_info_fn), "rb") as handle:
    tm_info = pickle.load(handle)

with open(tm_info["model_path"], "rb") as handle:
    lda_model = pickle.load(handle)

with open(os.path.join(path_to_popularities, per_year_fn), 'rb') as handle:
    lda_per_year = pickle.load(handle)

# with open(os.path.join(path_to_popularities, per_sentence_fn), 'rb') as handle:
#     lda_per_sentence = pickle.load(handle)

with open(os.path.join(path_to_popularities, per_abstract_fn), 'rb') as handle:
    lda_per_abstract = pickle.load(handle)

with open(os.path.join(path_to_rfl, rf_labelnames_dict_fn), "rb") as handle:
    rflabel_dict = pickle.load(handle)

tm_dictionary = gensim.corpora.Dictionary.load(tm_info["dic_path"])


for topic_cluster_name in topic_cluster_names:
    topic_cluster_id = tm_dictionary.token2id[topic_cluster_name[0]]
    topics = list()

    # Gather TM-Prob Information of Cluster
    for topic, comp in enumerate(lda_model.components_):
        norm_factor = np.sum(comp)
        tc_prob = comp[topic_cluster_id] / norm_factor
        topics.append((topic, tc_prob))

    years = sorted([year for year in lda_per_year.keys() if year_start <= year <= year_end])
    rf_labels = sorted(list(lda_per_year[years[0]].keys()))
    # num_topics = lda_per_year[years[0]][rf_labels[0]].shape[1]

    x = list()
    y_year = [list() for i in rf_labels]
    y_sent = [list() for i in rf_labels]
    y_abstract = [list() for i in rf_labels]

    for year in years:
        x.append(year)
        # num_sentences = sum([lda_per_sentence[year][rf].shape[0] for rf in rf_labels])
        num_abstracts = sum([lda_per_abstract[year][rf].shape[0] for rf in rf_labels])
        for rf in rf_labels:
            y_year[rf].append(sum([lda_per_year[year][rf][0][topic[0]]*topic[1]
                              for topic in topics]))

            # num_rf_sentences = lda_per_sentence[year][rf].shape[0]
            # y_sent[rf].append(sum([sum([lda_per_sentence[year][rf][n][topic[0]]*topic[1]
            #                             for n in range(num_rf_sentences)])
            #                        for topic in topics]) / num_sentences)

            num_rf_abstracts = lda_per_abstract[year][rf].shape[0]
            y_abstract[rf].append(sum([sum([lda_per_abstract[year][rf][n][topic[0]] * topic[1]
                                            for n in range(num_rf_abstracts)])
                                       for topic in topics]) / num_abstracts)

    all_color_cylce = ['darksalmon',
                       'sienna',
                       'sandybrown',
                       'bisquie',
                       'tan']

    color_cycle = all_color_cylce[:len(rf_labels)]

    plt.title(f"{topic_cluster_name[1]}")
    plt.stackplot(x, y_year,
                  labels=[rflabel_dict['id2label'][rf_id] for rf_id in rf_labels],
                  colors=color_cycle)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path_to_fig_save, f"{topic_cluster_name[1]}_topic_popularity_year_avg_{cluster_type}_{v_num_topics}.png"))
    #plt.show()
    plt.clf()

    # plt.title(f"{topic_cluster_name[1]}")
    # plt.stackplot(x, y_sent,
    #               labels=[rflabel_dict['id2label'][rf_id] for rf_id in rf_labels],
    #               colors=color_cycle)
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(path_to_fig_save, f"{topic_cluster_name[1]}_topic_popularity_sentence_{cluster_type}_{v_num_topics}.png"))
    # #plt.show()
    # plt.clf()

    plt.title(f"{topic_cluster_name[1]}")
    plt.stackplot(x, y_abstract,
                  labels=[rflabel_dict['id2label'][rf_id] for rf_id in rf_labels],
                  colors=color_cycle)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path_to_fig_save,
                             f"{topic_cluster_name[1]}_topic_popularity_abstract_{cluster_type}_{v_num_topics}.png"))
    # plt.show()
    plt.clf()