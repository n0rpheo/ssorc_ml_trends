import random
import pickle
from time import time
import numpy as np
from scipy.special import comb
from scipy.optimize import nnls

from joblib import Parallel, delayed

from utils.LoopTimer import LoopTimer
from MLRes.occ import distances


class occTopMention:
    def __init__(self, path):
        with open(path, "rb") as handle:
            occ = pickle.load(handle)

        self.mentions = occ["mentions"]
        self.m_strings = [self.mentions[i]["string"] for i in range(len(self.mentions))]
        self.clusters = occ["cluster_matrix"]
        self.similarity_matrix = occ["similarity_matrix"]

        self.rep2cid = dict()
        self.cid2rep = dict()

        for cid in range(self.clusters.shape[1]):
            if sum(self.clusters[:, cid]) > 0:
                m_ids = np.where(self.clusters[:, cid] == 1)[0]
                top_rep_rating = 0
                top_rep = m_ids[0]
                for v in m_ids:
                    distance = 0
                    for u in m_ids:
                        if v == u:
                            continue

                        d = self.similarity_matrix[v, u]
                        distance += d
                    if len(m_ids) > 1:
                        distance = distance / (len(m_ids) - 1)
                    else:
                        distance = 1
                    if distance > top_rep_rating:
                        top_rep = self.mentions[v]['string']
                        top_rep_rating = distance
                self.cid2rep[cid] = top_rep
                if top_rep not in self.rep2cid:
                    self.rep2cid[top_rep] = list()
                self.rep2cid[top_rep].append(cid)

    def get(self, test_string):
        if test_string.lower() in self.m_strings:
            mid = self.m_strings.index(test_string.lower())

            cluster_ids = np.where(self.clusters[mid, :] == 1)[0]
            top_reps = set()
            for cid in cluster_ids:
                top_rep_rating = 0
                top_rep = test_string
                m_ids = np.where(self.clusters[:, cid] == 1)[0]
                for v in m_ids:
                    rating = 0
                    for u in m_ids:
                        if v == u:
                            continue
                        r = self.similarity_matrix[v, u]
                        rating += r
                    if len(m_ids) > 1:
                        rating = rating / (len(m_ids) - 1)
                    else:
                        rating = 1
                    if rating > top_rep_rating:
                        top_rep = self.mentions[v]['string']
                        top_rep_rating = rating
                top_reps.add(top_rep)
            return list(top_reps)
        else:
            #print(f"out {test_string}")
            return [test_string]


def make_similarity_matrix(mentions,
                           nlp,
                           acronym_dict,
                           n_jobs=-1):
    num_mentions = len(mentions)

    """
        Compute Similarity Matrix
    """

    expanded_mentions = list()
    for mid, mention in enumerate(mentions):
        expanded_list = [mention['orth_with_ws'].copy()]
        distances.expand_mention(expanded_list, acronym_dict)

        expanded_mentions.append(expanded_list)

    """
        Removing Ignore_Words
        Compute and Store Vectors
    """

    expanded_mentions_vectors = list()
    expanded_mentions_vector_averages = list()

    lt = LoopTimer(update_after=200, avg_length=20000, target=len(expanded_mentions))
    for expm in expanded_mentions:
        list_strings = ["".join([token for token in tokens if token.strip().lower() not in distances.ignore_words]) for
                        tokens in expm]
        list_spacy = [nlp(string) for string in list_strings]
        m_vectors = [doc.vector for doc in list_spacy]
        avg_vec = [sum(m_vectors) / len(m_vectors)]
        expanded_mentions_vector_averages.append(avg_vec)
        expanded_mentions_vectors.append(m_vectors)
        lt.update("Calc Vectors")

    """
        Compute Vector-Similarity-Matrix
    """
    print()
    lt = LoopTimer(update_after=5, avg_length=1000, target=num_mentions)
    sim_matrix = list()
    for vid, v in enumerate(expanded_mentions_vectors):
        sub_sims = Parallel(n_jobs=n_jobs)(delayed(distances.occ_vec)(v, u) for u in expanded_mentions_vectors)
        lt.update("Calc Affinity-Matrix")
        sim_matrix.append(sub_sims)

    return np.array(sim_matrix)


def init_clusters(n_clusters, n_mentions, p):
    m_range = range(n_mentions)
    label_combs = [comb(n_clusters, p_, exact=True) for p_ in range(1, p + 1)]
    n_labels = sum(label_combs)

    labeling = dict()
    for mention in m_range:
        labeling[mention] = random.randint(0, n_labels - 1)

    clusters = list()
    for n_mention in m_range:
        label_id = labeling[n_mention]
        p = 0
        while label_id >= 0:
            label_id -= label_combs[p]
            p += 1
        label_id += label_combs[p - 1]
        assigned_clusters = list()
        calc_p = p
        while calc_p > 1:
            calc_p -= 1
            cluster_cp = int(label_id / label_combs[calc_p - 1])
            label_id -= cluster_cp * label_combs[calc_p - 1]
            assigned_clusters.append(cluster_cp)
        assigned_clusters.append(label_id)
        c_vec = [1 if c in assigned_clusters else 0 for c in range(n_clusters)]
        clusters.append(c_vec)
    return np.array(clusters)


def jac_coef(v, u, clusters):
    v_set = set(np.where(clusters[v, :] == 1)[0])
    u_set = set(np.where(clusters[u, :] == 1)[0])

    return len(v_set.intersection(u_set)) / len(v_set.union(u_set))


def cost_occ(sim_matrix, clusters):
    m_range = range(sim_matrix.shape[0])
    cocc_sum = 0
    for v in m_range:
        for u in m_range:
            if u == v:
                continue
            cocc_sum += abs(jac_coef(v, u, clusters) - sim_matrix[v, u])

    return cocc_sum*0.5


def optimize(init_cluster, sim_matrix, p, tol=0.00005):
    n_clusters = init_cluster.shape[1]
    n_mentions = init_cluster.shape[0]
    m_range = range(n_mentions)

    cluster = np.copy(init_cluster)

    nnls_t = 0
    build_matrix_t = 0
    get_best_solution_t = 0
    calc_cost_t = 0

    best_cocc = cost_occ(sim_matrix, init_cluster)

    print()
    print(f"Start Optimization with: {best_cocc}")

    init_line = np.ones(n_clusters + 1).reshape(1, n_clusters + 1)
    init_line[0][0] = -1
    ones_vector = np.ones((n_mentions, 1))

    lt = LoopTimer(update_after=1, avg_length=20000)

    best_cluster = np.concatenate(cluster)

    while True:
        for v in m_range:
            """
                Construct Matrix
            """
            bm_t_start = time()

            zj_vec = sim_matrix[v, :].reshape(n_mentions, 1)
            mid_matrix = cluster*(ones_vector+zj_vec)
            a_matrix = np.concatenate((np.concatenate((zj_vec, -mid_matrix), axis=1), init_line), axis=0)

            a_matrix = np.delete(a_matrix, v, axis=0)

            n_sj_vec = cluster.sum(1).reshape(n_mentions, 1) * zj_vec
            b_vec = np.concatenate((-n_sj_vec, np.array([[0]])), axis=0).reshape(n_mentions+1)
            b_vec = np.delete(b_vec, v, axis=0)

            bm_t_end = time()
            build_matrix_t += (bm_t_end - bm_t_start)

            """
                Non Negative Least Squares
            """
            nnls_t_start = time()
            nnls_result = nnls(a_matrix, b_vec)[0][1:]
            nnls_ind = np.argpartition(nnls_result, -p)[-p:]
            nnls_ind_sorted = nnls_ind[np.argsort(nnls_result[nnls_ind])][::-1]
            nnls_t_end = time()
            nnls_t += (nnls_t_end - nnls_t_start)

            """
                Retrieve best feasible solution 
            """
            gbs_t_start = time()
            min_dist = float("inf")
            best_sq = float("inf")
            for q in range(1, p+1):
                Sq_ = nnls_ind_sorted[0:q]
                sq = set(Sq_)
                dist = 0
                for j in m_range:
                    j_clusters = np.where(cluster[j, :] == 1)[0]
                    dist += abs(distances.occ_h(sq, j_clusters) - sim_matrix[v, j])

                if dist < min_dist:
                    min_dist = dist
                    best_sq = np.copy(Sq_)

            for i in range(n_clusters):
                cluster[v, i] = 1 if i in best_sq else 0

            gbs_t_end = time()
            get_best_solution_t += (gbs_t_end - gbs_t_start)

        cc_t_start = time()
        new_cocc = cost_occ(sim_matrix, cluster)
        cc_t_end = time()
        calc_cost_t += (cc_t_end-cc_t_start)

        sum_t = calc_cost_t + build_matrix_t + nnls_t + get_best_solution_t
        bmt = round((build_matrix_t / sum_t) * 100, 2)
        nnlst = round((nnls_t/sum_t)*100, 2)
        gbst = round((get_best_solution_t/sum_t) * 100, 2)
        cct = round((calc_cost_t / sum_t) * 100, 2)
        print()
        lt.update(f"Optimize: {best_cocc} -> {new_cocc} | Build Matrix: {bmt} % |  NNLS: {nnlst} % | GBS: {gbst} % | CCT: {cct} %")
        print()
        if abs(best_cocc - new_cocc) < tol or new_cocc > best_cocc:
            break
        best_cocc = new_cocc
        best_cluster = np.copy(cluster)
    print()
    print(f"End Optimization with: {best_cocc}")
    return best_cluster, best_cocc
