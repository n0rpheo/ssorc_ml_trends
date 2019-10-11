from __future__ import absolute_import, division, print_function
import os
import math
import pickle

import pandas as pd
import numpy as np

from spacy.vocab import Vocab
from spacy.tokens import Doc

from tensorflow import keras

from abstract_parser.doc2vec_features import doc_to_feature_vector

from utils.LoopTimer import LoopTimer
from info import paths

nlp_model = "en_newmodel"

# Output
aid_list_fn_out = "aid_list.pickle"
rf_list_fn_out = "rf_list.pickle"

# Infos for the Abstract Parser
rf_model_fn = f"ann_model.h5"

# All Paths
path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)
path_to_pandas = os.path.join(paths.to_root, "pandas", nlp_model)
path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)

# RF Model
rf_model = keras.models.load_model(os.path.join(path_to_rfl, rf_model_fn))

aid_list = list()
rf_list = list()

print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
db_size = len(infoDF)

"""
    Abstract Parser
    Assign a rhetorical function to every sentence of every abstract
"""

lc = LoopTimer(update_after=1, avg_length=200, target=db_size)
for idx, (abstract_id, df_row) in enumerate(infoDF.iterrows()):

    doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
    feature_vector = np.array(doc_to_feature_vector(doc))

    if any(entry is None for entry in feature_vector):
        breaker = lc.update(f"Abstract Parser - BAD FV - {len(aid_list)}")
        continue

    prediction_distr = rf_model.predict(feature_vector)
    prediction = [pred.argmax() for pred in prediction_distr]

    aid_list.append(abstract_id)
    rf_list.append(prediction)

    breaker = lc.update(f"Abstract Parser - {len(aid_list)}")


if not os.path.isdir(path_to_popularities):
    print(f"Create Directory {path_to_popularities}")
    os.mkdir(path_to_popularities)

with open(os.path.join(path_to_popularities, aid_list_fn_out), 'wb') as handle:
    pickle.dump(aid_list, handle)

with open(os.path.join(path_to_popularities, rf_list_fn_out), 'wb') as handle:
    pickle.dump(rf_list, handle)
