from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

import sklearn.metrics as skmetrics
import numpy as np

from prettytable import PrettyTable


def check_string_for_english(input_text):
    try:
        detection = detect_langs(input_text)

        result = False
        for lang in detection:
            if lang.lang == 'en':
                if lang.prob > 0.9:
                    result = True

        return result
    except LangDetectException:
        return False


class Scoring:
    def __init__(self, targets, prediction, avg_type="weighted", target_dic=None):
        self.targets = targets
        self.prediction = prediction
        self.avg_type = avg_type
        self.target_dic = target_dic

    def print(self):
        labels = list(set(self.targets))

        conf_matrix = skmetrics.confusion_matrix(self.targets, self.prediction, labels=labels)
        prec_score = skmetrics.precision_score(self.targets, self.prediction, labels=labels, average=self.avg_type)
        accuracy_score = skmetrics.accuracy_score(self.targets, self.prediction)
        recall_score = skmetrics.recall_score(self.targets, self.prediction, average=self.avg_type)
        f1_score = skmetrics.f1_score(self.targets, self.prediction, average=self.avg_type)

        print_matrix = list()

        for label in labels:
            label_idx = labels.index(label)

            rel = np.sum(conf_matrix, axis=1)[label_idx]
            ret = np.sum(conf_matrix, axis=0)[label_idx]

            precision = conf_matrix[label_idx][label_idx] / ret if ret > 0 else 0
            recall = conf_matrix[label_idx][label_idx] / rel

            if self.target_dic is not None:
                label_name = self.target_dic[label]
            else:
                label_name = label

            print_matrix.append([label_name, ret, rel, str(precision)[0:5], str(recall)[0:5]])

        p = PrettyTable()
        p.field_names = ["Label", "Retrieved", "Relevant", "Precision", "Recall"]

        for row in print_matrix:
            p.add_row(row)

        print(p.get_string(header=True, border=True))
        print()

        p2 = PrettyTable()
        p2.add_row(["Accuracy", str(accuracy_score)[0:5]])
        p2.add_row(["Precision", str(prec_score)[0:5]])
        p2.add_row(["Recall", str(recall_score)[0:5]])
        p2.add_row(["F1-Score", str(f1_score)[0:5]])

        print(p2.get_string(header=False, border=True))

    def get_scores(self):
        labels = list(set(self.targets))
        prec_score = skmetrics.precision_score(self.targets, self.prediction, labels=labels, average=self.avg_type)
        accuracy_score = skmetrics.accuracy_score(self.targets, self.prediction)
        recall_score = skmetrics.recall_score(self.targets, self.prediction, average=self.avg_type)
        f1_score = skmetrics.f1_score(self.targets, self.prediction, average=self.avg_type)

        return prec_score, accuracy_score, recall_score, f1_score

    def save(self, path, title, open_mode='w'):
        labels = list(set(self.targets))
        precision = dict()
        recall = dict()
        rel_dict = dict()
        ret_dict = dict()

        conf_matrix = skmetrics.confusion_matrix(self.targets, self.prediction, labels=labels)
        prec_score = skmetrics.precision_score(self.targets, self.prediction, labels=labels, average=self.avg_type)
        accuracy_score = skmetrics.accuracy_score(self.targets, self.prediction)
        recall_score = skmetrics.recall_score(self.targets, self.prediction, average=self.avg_type)
        f1_score = skmetrics.f1_score(self.targets, self.prediction, average=self.avg_type)

        for label in labels:
            label_idx = labels.index(label)

            rel = np.sum(conf_matrix, axis=1)[label_idx]
            ret = np.sum(conf_matrix, axis=0)[label_idx]

            precision[label] = conf_matrix[label_idx][label_idx] / ret if ret > 0 else 0
            recall[label] = conf_matrix[label_idx][label_idx] / rel

            rel_dict[label] = rel
            ret_dict[label] = ret

        if open_mode == 'a':
            write_string = "\n\n\n"
        else:
            write_string = ""

        write_string += f"{title}\n\n"

        write_string += f"\tLabel\tRetrieved\tRelevant\tPrecision\tRecall\n"
        for label in labels:
            write_string += f"\t{str(label)[0:5]}\t{str(ret_dict[label])[0:5]}\t{str(rel_dict[label])[0:5]}\t"
            write_string += f"{str(precision[label])[0:5]}\t{str(recall[label])[0:5]}\n"

        write_string += f"\n"
        write_string += f"\tAccuracy\t{str(accuracy_score)[0:5]}\n"
        write_string += f"\tPrecision\t{str(prec_score)[0:5]}\n"
        write_string += f"\tRecall\t{str(recall_score)[0:5]}\n"
        write_string += f"\tF1-Score\t{str(f1_score)[0:5]}\n"

        with open(path, mode=open_mode) as write_file:
            write_file.write(write_string)
