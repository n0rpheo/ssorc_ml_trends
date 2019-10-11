import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from gensim.corpora.dictionary import Dictionary

import topic_modeling.lda_features as ldaf

from info import paths


sns.set()

nlp_model = "en_newmodel"
dic_name = "lemma_dict_p2.gendic"
corpus_name = "corpus_p2.pickle"

path_to_pandas = os.path.join(paths.to_root, "pandas", nlp_model)

dictionary = Dictionary.load(os.path.join(path_to_pandas, dic_name))
with open(os.path.join(path_to_pandas, corpus_name), "rb") as handle:
    corpus = pickle.load(handle)

lemma_d_list = corpus['lemma_document']  # [:10000]
docs = ldaf.docs_preprocessor(lemma_d_list)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Set parameters.
num_topics = 5
chunksize = 500
passes = 20
iterations = 100
eval_every = 1

# Make a index to word dictionary.
temp = dictionary[0]  # only to "load" the dictionary.
id2word = dictionary.id2token

limit = 1500
start = 10
step = 50
model_list, coherence_values = ldaf.compute_coherence_values(dictionary=dictionary,
                                                             corpus=corpus,
                                                             texts=docs,
                                                             start=start,
                                                             limit=limit,
                                                             step=step)

sns.set()
# Show graph

x = range(start, limit, step)

fig, ax = plt.subplots()
ax.plot(x, coherence_values, label='Coherence Value')
ax.legend(loc='best')
ax.set_xlabel("Num Topics")
ax.set_ylabel("Coherence Score")
plt.show()