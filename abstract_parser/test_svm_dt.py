import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from utils.LoopTimer import LoopTimer
from utils.functions import Scoring

from info import paths

names = [
    # "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Neural Net"
]

svclin_p = [0.01, 0.1, 1]
svcrbf_p = [0.01, 0.1, 1]
dt_p = [3, 5, 10, 50]
mlp_p = [0.01, 0.1, 1]

classifiers = [
    [(f"SVC-Lin @ {p}", SVC(kernel="linear", C=p, decision_function_shape='ovo')) for p in svclin_p],
    # [(f"SVC-RBF @ {p}", SVC(kernel='rbf', gamma='auto', C=p, decision_function_shape='ovo')) for p in svcrbf_p],
    [(f"DT @ {p}", DecisionTreeClassifier(max_depth=p)) for p in dt_p],
]

clfs = [element for sublist in classifiers for element in sublist]

"""

"""
nlp_model = "en_newmodel"

allow_tw = False
feature_type = "ngram"  # or "svec"

feature_info_name = f"doc2vec_{'tw' if allow_tw else 'notw'}_features_info.pickle"
gold_label_name = f"gold_labels_{'tw' if allow_tw else 'notw'}.pickle"

"""
"""

path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)

with open(os.path.join(path_to_rfl, feature_info_name), "rb") as handle:
    feature_dict = pickle.load(handle)

with open(os.path.join(path_to_rfl, gold_label_name), "rb") as handle:
    gold_label_dict = pickle.load(handle)

all_features = feature_dict[f"features_{feature_type}"]
all_targets = feature_dict["targets"]

all_gold_features = gold_label_dict[f"features_{feature_type}"]
all_gold_targets = gold_label_dict["targets"]

all_features = np.array(all_features)
all_targets = np.array(all_targets)
print(all_features.shape)
print(all_targets.shape)
unique, counts = np.unique(all_targets, return_counts=True)
print(dict(zip(unique, counts)))

all_features, _, all_targets, _ = train_test_split(all_features,
                                                   all_targets,
                                                   test_size=0.1,
                                                   random_state=0,
                                                   shuffle=True)

n_labels = len(set(all_targets))
n_inputs = all_features.shape[1]
print(all_features.shape)
print(all_targets.shape)
unique, counts = np.unique(all_targets, return_counts=True)
print(dict(zip(unique, counts)))


label2id = dict()
id2label = dict()
all_int_targets = list()
for idx, entry in enumerate(all_targets):
    if entry not in label2id:
        label2id[entry] = len(label2id)
        id2label[len(id2label)] = entry
    all_int_targets.append(label2id[entry])

all_gold_int_targets = list()
for idx, entry in enumerate(all_gold_targets):
    if entry not in label2id:
        label2id[entry] = len(label2id)
        id2label[len(id2label)] = entry
    all_gold_int_targets.append(label2id[entry])

all_int_targets = np.array(all_int_targets)
all_gold_int_targets = np.array(all_gold_int_targets)

rflabel_dict = {"id2label": id2label,
                "label2id": label2id}

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_int_targets,
                                                                                            test_size=0.2,
                                                                                            random_state=42,
                                                                                            shuffle=True)

print(f"Learning Feature-Vector-Shape: {learning_features.shape}")
print(f"Holdback Feature-Vector-Shape: {holdback_features.shape}")
result_list = list()

print()
lc = LoopTimer(update_after=1, avg_length=1, target=len(clfs))
print(f"Feature Type: {feature_type} | Allow TW: {allow_tw}")
for name, clf in clfs:
    lc.update(f"{name} starting")
    clf.fit(learning_features, learning_targets)
    prediction = clf.predict(holdback_features)
    gold_prediction = clf.predict(all_gold_features)
    print(f"{name}:")
    print("----------")
    scoring = Scoring(holdback_targets, prediction, target_dic=rflabel_dict['id2label'])
    scoring.print()
    print()
    print("Gold Label Test")
    scoring = Scoring(all_gold_int_targets, gold_prediction, target_dic=rflabel_dict['id2label'])
    scoring.print()
    print("----------")
    print("----------")
    print("----------")
    print()
    print()
