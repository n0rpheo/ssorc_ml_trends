import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gensim
import pandas as pd

from utils.LoopTimer import LoopTimer

from info import paths

sns.set()

nlp_model = "en_newmodel"

#Output

# RF Labels

rf_labelnames_dict_fn = f"ann_model_labelnames.dict"

# Topic Cluster
cluster_type = "p2"
v_num_topics = 500

if cluster_type == "p1":
    """
        Cluster P1
    """

    per_year_fn = "pop_per_year.pickle"
    # per_sentence_fn = "pop_per_sentence.pickle"
    tm_info_fn = "lemma_vec_occ1_300.info"
    topic_cluster_names = []
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

topic_cluster_name_dict = dict()
for tcn in topic_cluster_names:
    topic_cluster_name_dict[tcn[0]] = tcn[1]

# Year Range
year_start = 1980
year_end = 2017

# All Paths
path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)
path_to_fig_save = "/home/norpheo/Documents/thesis/Results/fig/popularities/new_model/"

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

topic_cluster_ids = [tm_dictionary.token2id[topic_cluster_name[0]] for topic_cluster_name in topic_cluster_names]

years = sorted([year for year in lda_per_year.keys() if year_start <= year <= year_end])
rf_labels = sorted(list(lda_per_year[years[0]].keys()))
num_topics = lda_per_year[years[0]][rf_labels[0]].shape[1]

# Gather TM-Prob Information of Cluster

topics_probs = dict()
for topic_cluster_id in topic_cluster_ids:
    topics = list()
    for topic, comp in enumerate(lda_model.components_):
        norm_factor = np.sum(comp)
        tc_prob = comp[topic_cluster_id] / norm_factor
        topics.append((topic, tc_prob))
    topics_probs[topic_cluster_id] = topics

x = list()
for year in years:
    x.append(year)

y_topics = dict()
y_topics_sentences = dict()
y_topics_abstracts = dict()
lc = LoopTimer(update_after=1, avg_length=100, target=len(topic_cluster_ids)*len(years))
for topic_cluster_id in topic_cluster_ids:

    topic_probs = topics_probs[topic_cluster_id]
    y_topics[topic_cluster_id] = list()
    y_topics_sentences[topic_cluster_id] = list()
    y_topics_abstracts[topic_cluster_id] = list()

    for year in years:
        # num_sentences = sum([lda_per_sentence[year][rf].shape[0] for rf in rf_labels])
        num_abstracts = sum([lda_per_abstract[year][rf].shape[0] for rf in rf_labels])

        y_topics[topic_cluster_id].append(
            sum(  # over RF
                [sum(  # Over Cluster-Topics
                    [lda_per_year[year][rf][0][topic_prob[0]] * topic_prob[1] for topic_prob in topic_probs]
                ) for rf in rf_labels]
            )
        )

        # y_topics_sentences[topic_cluster_id].append(
        #     sum(  # Over RF
        #         [sum(  # Over Cluster-Topics
        #             [sum(  # Over Sentences
        #                 [lda_per_sentence[year][rf][n][topic_prob[0]] * topic_prob[1] for n in range(lda_per_sentence[year][rf].shape[0])]
        #                  ) for topic_prob in topic_probs]
        #         ) for rf in rf_labels]) / num_sentences)

        y_topics_abstracts[topic_cluster_id].append(
            sum(  # Over RF
                [sum(  # Over Cluster-Topics
                    [sum(  # Over Abstracts
                        [lda_per_abstract[year][rf][n][topic_prob[0]] * topic_prob[1] for n in
                         range(lda_per_abstract[year][rf].shape[0])]
                    ) for topic_prob in topic_probs]
                ) for rf in rf_labels]) / num_abstracts)
        lc.update(f"Build - {topic_cluster_id} - {year}")

print()
cluster_df = pd.DataFrame.from_dict(y_topics)
cluster_df_normalized = cluster_df.div(cluster_df.sum(axis=1), axis=0)

sentence_cluster_df = pd.DataFrame.from_dict(y_topics_sentences)
sentence_cluster_df_normalized = sentence_cluster_df.div(sentence_cluster_df.sum(axis=1), axis=0)

abstract_cluster_df = pd.DataFrame.from_dict(y_topics_abstracts)
abstract_cluster_df_normalized = abstract_cluster_df.div(abstract_cluster_df.sum(axis=1), axis=0)

y_sentence = np.transpose(sentence_cluster_df.values)
y_sentence_norm = np.transpose(sentence_cluster_df_normalized.values)

y = np.transpose(cluster_df.values)
y_norm = np.transpose(cluster_df_normalized.values)

y_abstract = np.transpose(abstract_cluster_df.values)
y_abstract_norm = np.transpose(abstract_cluster_df_normalized.values)

all_color_cylce = ['lawngreen',
                   'forestgreen',
                   'navy',
                   'deepskyblue',
                   'sienna',
                   'sandybrown',
                   'brown',
                   'yellow',
                   'lightyellow',
                   'violet',
                   'darkorchid'
                   ]

color_cycle = all_color_cylce[:y.shape[0]]

"""
    Per Year
"""

plt.title(f"ML Topic Trajectories - Normalized")
plt.stackplot(x, y_norm,
              labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
                      for topic_cluster_id in topic_cluster_ids],
              colors=color_cycle)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_norm_{cluster_type}_{v_num_topics}.png"),
            bbox_inches="tight")
# plt.show()

plt.clf()

plt.title(f"ML Topic Trajectories")
plt.stackplot(x, y,
              labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
                      for topic_cluster_id in topic_cluster_ids],
              colors=color_cycle)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_{cluster_type}_{v_num_topics}.png"),
            bbox_inches="tight")
# plt.show()

# """
#     Per Sentence
# """
#
# plt.clf()
#
# plt.title(f"ML Topic Trajectories - Normalized")
# plt.stackplot(x, y_sentence_norm,
#               labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
#                       for topic_cluster_id in topic_cluster_ids],
#               colors=color_cycle)
# plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
# plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_sentence_norm_{cluster_type}_{v_num_topics}.png"),
#             bbox_inches="tight")
# # plt.show()
#
# plt.clf()
#
# plt.title(f"ML Topic Trajectories")
# plt.stackplot(x, y_sentence,
#               labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
#                       for topic_cluster_id in topic_cluster_ids],
#               colors=color_cycle)
# plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
# plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_sentence_{cluster_type}_{v_num_topics}.png"),
#             bbox_inches="tight")
# # plt.show()

"""
    Per Abstract
"""

plt.clf()

plt.title(f"ML Topic Trajectories - Normalized")
plt.stackplot(x, y_abstract_norm,
              labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
                      for topic_cluster_id in topic_cluster_ids],
              colors=color_cycle)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_abstract_norm_{cluster_type}_{v_num_topics}.png"),
            bbox_inches="tight")
# plt.show()

plt.clf()

plt.title(f"ML Topic Trajectories")
plt.stackplot(x, y_abstract,
              labels=[topic_cluster_name_dict[tm_dictionary[topic_cluster_id]]
                      for topic_cluster_id in topic_cluster_ids],
              colors=color_cycle)
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(path_to_fig_save, f"all_topic_popularity_abstract_{cluster_type}_{v_num_topics}.png"),
            bbox_inches="tight")
# plt.show()