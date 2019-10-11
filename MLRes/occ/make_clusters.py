import pickle
import os
from joblib import Parallel, delayed
import math
import spacy
import numpy as np

import MLRes.occ.occ as occ
from info import paths


nlp_model = "en_newmodel"
p = 2    # maximum number of clusters a mentions can be assigned to
cluster_file_name = f"occ_p{p}_best.pickle"
sim_file_name = "simimlarities.npy"
force_sim_calc = False

path_to_mlgenome = os.path.join(paths.to_root, "mlgenome", nlp_model)
nlp_path = os.path.join(paths.to_root, "models", nlp_model)
path_to_sim_file = os.path.join(path_to_mlgenome, sim_file_name)

with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

with open(os.path.join(path_to_mlgenome, "ml_acronyms.pickle"), "rb") as handle:
    acronym_dict = pickle.load(handle)

mentions = mentions

print(len(mentions))

if os.path.isfile(path_to_sim_file) and not force_sim_calc:
    sim_matrix = np.load(path_to_sim_file)
else:
    print("Loading spaCy Vocab...")
    nlp = spacy.load(nlp_path)

    sim_matrix = occ.make_similarity_matrix(mentions=mentions,
                                            nlp=nlp,
                                            acronym_dict=acronym_dict)
    np.save(path_to_sim_file, sim_matrix)


cluster_matrices = [occ.init_clusters(n_clusters=len(mentions), n_mentions=len(mentions), p=p) for i in range(40)]

best_solution = Parallel(n_jobs=12)(delayed(occ.optimize)(cluster_matrix, sim_matrix, 1)
                                    for cluster_matrix in cluster_matrices)

best_score = math.inf
best_cluster = None

for idx, bs in enumerate(best_solution):
    print(bs[1])

    if bs[1] < best_score:
        best_score = bs[1]
        best_cluster = bs[0]

print()
print(f"Best Score: {best_score}")
print(best_cluster.shape)

print(f"Save to {cluster_file_name}")
save_dict = {"mentions": mentions,
             "cluster_matrix": best_cluster,
             "similarity_matrix": sim_matrix}

with open(os.path.join(path_to_mlgenome, cluster_file_name), 'wb') as handle:
    pickle.dump(save_dict, handle)
