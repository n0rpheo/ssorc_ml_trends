import os
from sklearn.decomposition import LatentDirichletAllocation
import scipy.sparse
import pickle

from info import paths


"""
 #
 # Infos
 #
"""

nlp_model = "en_newmodel"
n_topics = 500
info_file_name = "lemma_vec_occ2.info"

suffix = ""

iterations = 20

doc_topic_prior = 1 / n_topics
topic_word_prior = 1 / n_topics

"""
 #
 #
 #
"""

path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
path_to_model = os.path.join(path_to_tm, "model")
if not os.path.isdir(path_to_model):
    print(f"Create Directory {path_to_model}")
    os.mkdir(path_to_model)

info_name_output = f"{info_file_name[:-5]}_{n_topics}{suffix}.info"
model_file_name = f"{info_file_name[:-5]}_{n_topics}topics{suffix}.model"
with open(os.path.join(path_to_tm, info_file_name), "rb") as handle:
    info = pickle.load(handle)

info['model_path'] = os.path.join(path_to_model, model_file_name)

tm_features = scipy.sparse.load_npz(info['feature_path'])

print(tm_features.shape)

lda_model = LatentDirichletAllocation(n_components=n_topics,
                                      doc_topic_prior=doc_topic_prior,
                                      topic_word_prior=topic_word_prior,
                                      random_state=0,
                                      learning_method='batch',
                                      verbose=1,
                                      max_iter=iterations)
print(f"Training Model. - {info_file_name[:-5]} - {n_topics}")
lda_model.fit(tm_features)

with open(info['model_path'], "wb") as handle:
    pickle.dump(lda_model, handle)

with open(os.path.join(path_to_tm, info_name_output), "wb") as handle:
    pickle.dump(info, handle)

