import os
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pandas as pd

from utils.LoopTimer import LoopTimer
from info import paths

"""
==================================
"""
old_nlp_model = 'en_core_web_lg'
"""
==================================
"""

"""
NLP with Trained NER
==================================
"""
nlp_model = "en_newmodel"
nlp_path = os.path.join(paths.to_root, "models", nlp_model)
"""
==================================
"""

path_to_annotations = os.path.join(paths.to_root, "annotations_version", nlp_model)
path_to_old_annotations = os.path.join(paths.to_root, "annotations_version", old_nlp_model)

if not os.path.isdir(path_to_annotations):
    print(f"Create Directory {path_to_annotations}")
    os.mkdir(path_to_annotations)

print("Load Vocab and NLP...")
nlp = spacy.load(nlp_path)
old_vocab = Vocab().from_disk(os.path.join(path_to_old_annotations, "spacy.vocab"))
old_infoDF = pd.read_pickle(os.path.join(path_to_old_annotations, 'info_db.pandas'))

print("Starting")
lt = LoopTimer(update_after=10, avg_length=1000, target=len(old_infoDF))
for abstract_id, row in old_infoDF.iterrows():
    file_path = os.path.join(path_to_old_annotations, f"{abstract_id}.spacy")
    old_doc = Doc(old_vocab).from_disk(file_path)
    abstract = old_doc.text
    doc = nlp(abstract)

    doc.to_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
    lt.update("Re-Preprocess")

nlp.vocab.to_disk(os.path.join(path_to_annotations, "spacy.vocab"))
print(f"Vocab Size: {len(nlp.vocab)}")

nlp.vocab.to_disk(os.path.join(path_to_annotations, "spacy.vocab"))
old_infoDF.to_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))