from __future__ import absolute_import, division, print_function

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np

from tensorflow import keras

import matplotlib.pyplot as plt

from utils.functions import Scoring
from info import paths

"""

"""
nlp_model = "en_newmodel"
feature_type = "svec"  # or "ngram"
allow_tw = False
feature_info_name = f"doc2vec_{'tw' if allow_tw else 'notw'}_features_info.pickle"
gold_label_name = f"gold_labels_{'tw' if allow_tw else 'notw'}.pickle"
model_file_name = f"ann_model.h5"

epochs = 4

"""
"""
print(feature_info_name)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)


with open(os.path.join(path_to_rfl, feature_info_name), "rb") as handle:
    feature_dict = pickle.load(handle)

with open(os.path.join(path_to_rfl, gold_label_name), "rb") as handle:
    gold_label_dict = pickle.load(handle)

all_features = np.array(feature_dict[f"features_{feature_type}"])
all_targets = feature_dict["targets"]

all_gold_features = gold_label_dict[f"features_{feature_type}"]
all_gold_targets = gold_label_dict["targets"]

print(all_features.shape)
print(all_targets.shape)
unique, counts = np.unique(all_targets, return_counts=True)
print(dict(zip(unique, counts)))
n_labels = len(set(all_targets))
n_inputs = all_features.shape[1]

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

rflabel_dict = {"id2label": id2label,
                "label2id": label2id}

with open(os.path.join(path_to_rfl, f"{model_file_name[:-3]}_labelnames.dict"), "wb") as handle:
    pickle.dump(rflabel_dict, handle)

all_int_targets = np.array(all_int_targets)
all_gold_int_targets = np.array(all_gold_int_targets)

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_int_targets,
                                                                                            test_size=0.2,
                                                                                            random_state=42,
                                                                                            shuffle=True)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(learning_targets),
                                                  learning_targets)


print(f"Learning Feature-Vector-Shape: {learning_features.shape}")
print(f"Holdback Feature-Vector-Shape: {holdback_features.shape}")
print(f"Class Weights: {class_weights}")

"""
    TRAINING ANN
"""

model = keras.Sequential()

"""
model = keras.Sequential([
    keras.layers.Dense(n_inputs, activation=tf.nn.relu),
    keras.layers.Dense(n_labels, activation=tf.nn.softmax)
])
"""
# Input Layer
model.add(keras.layers.Dense(n_inputs, activation="relu", input_shape=(n_inputs, )))
# Hidden Layers

model.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(keras.layers.Dense(64, activation="relu"))

# Output Layer
model.add(keras.layers.Dense(n_labels, activation="softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Start fitting")
history = model.fit(learning_features,
                    learning_targets,
                    epochs=epochs,
                    class_weight=class_weights,
                    batch_size=100,
                    validation_data=(holdback_features, holdback_targets),
                    verbose=1)

print("Model fittet")

classifier = dict()
classifier["model"] = model

#print("\n\n")
#print('Save Model')
#with open(os.path.join(path_to_rfl, model_file_name), "wb") as model_file:
#    pickle.dump(classifier, model_file)

if holdback_features.shape[0] > 0:
    print(f"Feature Type: {feature_type} | Allow TW: {allow_tw}")
    print('Test Model Holdback')
    prediction = model.predict(holdback_features)

    prediction = np.array([pred.argmax() for pred in prediction])

    scores = Scoring(holdback_targets, prediction, target_dic=rflabel_dict['id2label'])
    scores.print()

    print()
    print('Test Model Goldlabels')
    gold_prediction = model.predict(all_gold_features)

    gold_prediction = np.array([pred.argmax() for pred in gold_prediction])

    scores = Scoring(all_gold_int_targets, gold_prediction, target_dic=rflabel_dict['id2label'])
    scores.print()

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

print("Model Saved")
model.save(os.path.join(path_to_rfl, model_file_name))

