import json
import os
import random
import logging
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from spacy.gold import GoldParse
from spacy.util import minibatch
import spacy
from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm

from info import paths


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                # only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    # dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1, label))

            training_data.append((text, {"entities": entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


def convert_dsv_to_spacy(dsv_file_path):
    with open(dsv_file_path, "rb") as handle:
        training_data = pickle.load(handle)
    return training_data


def test_data(test_docs, golds):
    for doc_to_test, gold in zip(test_docs, golds):
        d = {}

        for ent in doc_to_test.ents:
            d[ent.label_] = [0, 0, 0, 0, 0, 0]

        for ent in doc_to_test.ents:
            y_true = [ent.label_ if ent.label_ in x else f"Not {ent.label_}" for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ == ent.label_ else "Not {ent.label_}" for x in doc_to_test]
            if (d[ent.label_][0] == 0):
                # f.write("For Entity "+ent.label_+"\n")
                # f.write(classification_report(y_true, y_pred)+"\n")
                (p, r, f, s) = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                a = accuracy_score(y_true, y_pred)
                d[ent.label_][0] = 1
                d[ent.label_][1] += p
                d[ent.label_][2] += r
                d[ent.label_][3] += f
                d[ent.label_][4] += a
                d[ent.label_][5] += 1
    for i in d:
        print("\n For Entity " + i + "\n")
        print(f"Accuracy : {d[i][4] / d[i][5]}")
        print(f"Precision : {d[i][1] / d[i][5]}")
        print(f"Recall : {d[i][2] / d[i][5]}")
        print(f"F-score : {d[i][3] / d[i][5]}")


def train_spacy(TRAIN_DATA, epochs=10, sizes=1, model_name=None, test_size=0.0):
    print(f"Train-Data Shape: {len(TRAIN_DATA)}")

    iteration_list = list()
    tr_loss_list = list()

    if test_size == 0.0:
        val_data = TRAIN_DATA
        tr_data = TRAIN_DATA
    else:
        tr_data, val_data, _, _ = train_test_split(TRAIN_DATA, TRAIN_DATA,
                                                   random_state=2019, test_size=test_size)

    model = "en_core_web_lg"
    nlp = spacy.load(model)

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # Prepare Goldlabels
    golds = list()
    test_docs = list()
    for text, annot in val_data:
        doc_to_test = nlp(text)
        doc_gold_text = nlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))

        golds.append(gold)
        test_docs.append(doc_to_test)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in trange(epochs, desc="Epoch"):
            """
                Training
            """
            random.shuffle(tr_data)
            tr_losses = {}
            nb_tr_examples = 0
            nb_tr_steps = 0

            batches = minibatch(tr_data, size=sizes)
            for batch in tqdm(batches, desc="Batch-Train"):
                n_batch = len(batch)
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,
                    losses=tr_losses,
                )

                nb_tr_examples += n_batch
                nb_tr_steps += 1

            tr_loss_list.append(tr_losses['ner'] / nb_tr_steps)
            iteration_list.append(itn)

    # test the model and evaluate it
    test_data(test_docs, golds)

    if model_name is not None:
        # save model to output directory
        output_dir = os.path.join(paths.to_root, "models", model_name)
        nlp.meta["name"] = model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


data_file_name = "mlalgo_spacy_traindata.pickle"

output_model_name = "en_newmodel"

# dt_train_data = convert_dataturks_to_spacy("path")
dsv_train_data = convert_dsv_to_spacy(os.path.join(paths.to_root, "NER", data_file_name))

train_spacy(dsv_train_data, epochs=10, sizes=1, model_name=output_model_name)