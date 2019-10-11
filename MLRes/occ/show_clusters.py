import pickle
import os
import numpy as np

from info import paths


def write_node_file(path, ids, mention_list):
    with open(path, "w") as handle:
        handle.write("id;label\n")
        for m_id in ids:
            handle.write(f"{m_id};{mention_list[m_id]['string']}\n")


def write_edge_file(path, ids, simmatrix):
    with open(path, "w") as handle:
        handle.write("Source;Target;Weight\n")
        for v in ids:
            for u in ids:
                if u != v and simmatrix[u, v] > 0.75:
                    handle.write(f"{v};{u};{simmatrix[u, v]}\n")


"""
    Cluster Analysis
"""

nlp_model = "en_newmodel"

path_to_mlgenome = os.path.join(paths.to_root, "mlgenome", nlp_model)
with open(os.path.join(path_to_mlgenome, "occ_p2_best.pickle"), "rb") as handle:
    occ_res = pickle.load(handle)

csv_dir = "/home/norpheo/Documents/thesis/occ_csv"  # Destination for CSV (used in gephi)

mentions = occ_res["mentions"]
cluster_matrix = occ_res["cluster_matrix"]
sim_matrix = occ_res["similarity_matrix"]

nc = cluster_matrix.shape[1]
print(np.count_nonzero([sum(cluster_matrix[:, cid_]) for cid_ in range(nc)]))
all_ids = [i for i in range(len(mentions))]
c_list = list()
c_set = set()


write_node_file(os.path.join(csv_dir, "all_nodes.csv"), all_ids, mentions)
write_edge_file(os.path.join(csv_dir, "all_edges.csv"), all_ids, sim_matrix)

for cid in range(cluster_matrix.shape[1]):
    if sum(cluster_matrix[:, cid]) > 0:
        m_ids = np.where(cluster_matrix[:, cid] == 1)[0]
        top_rep_rating = 0
        top_rep = None
        for v in m_ids:
            distance = 0
            for u in m_ids:
                if v == u:
                    continue

                d = sim_matrix[v, u]
                distance += d
            if len(m_ids) > 1:
                distance = distance / (len(m_ids)-1)
            else:
                distance = 1
            if distance > top_rep_rating:
                top_rep = mentions[v]['string']
                top_rep_rating = distance
        if top_rep_rating > 0.7 and len(m_ids) > 10:
            c_list.append(top_rep)
            c_set.add(top_rep)
        print(f"Cluster {cid}: {top_rep} ({top_rep_rating}) {[mentions[i]['string'] for i in m_ids]}")
        print()
        #write_node_file(os.path.join(csv_dir, f"{cid}_{top_rep}_nodes.csv"), m_ids, mentions)
        #write_edge_file(os.path.join(csv_dir, f"{cid}_{top_rep}_edges.csv"), m_ids, sim_matrix)
        print("----------")
