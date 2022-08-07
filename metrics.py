import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class MultiMetrics(object):
    def __init__(self):
        self.recall = 0.0
        self.precision = 0.0
        self.f1 = 0.0
        self.num = 0

    def upd(self, predict, target):
        if torch.is_tensor(predict):
            predict = predict.data.cpu().numpy()
        predict = np.argmax(predict, axis=1)
        if torch.is_tensor(target):
            target = target.data.cpu().numpy().astype(np.long)

        predict = predict.ravel()
        target = target.ravel()
        #         print(predict.shape, target.shape)
        f1 = f1_score(target, predict, average='macro')
        p = precision_score(target, predict, average='macro')
        r = recall_score(target, predict, average='macro')
        #         print(f1, p, r)

        self.recall += r
        self.precision += p
        self.f1 += f1
        self.num += 1

    def get(self):
        metrics = {
            "Recall": self.recall / self.num,
            "Precision": self.precision / self.num,
            "F1-score": _f1_score(self.recall / self.num, self.precision / self.num),
        }
        return metrics


class Metrics(object):
    def __init__(self):
        self.recall = 0.0
        self.precision = 0.0
        self.specificity = 0.0
        self.acc = 0.0
        self.f1_score = 0.0
        self.jac = 0.0
        self.num = 0

    def upd(self, predict, target):
        if torch.is_tensor(predict):
            predict = predict.data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()

        predict = np.atleast_1d(predict.astype(np.bool))
        target = np.atleast_1d(target.astype(np.bool))

        tp = np.count_nonzero(predict & target)
        fn = np.count_nonzero(~predict & target)
        fp = np.count_nonzero(predict & ~target)
        tn = np.count_nonzero(~predict & ~target)

        self.recall += _recall(tp, fn)
        self.precision += _precision(tp, fp)
        self.specificity += _specificity(tn, fp)
        self.acc += _accuracy(tp, fp, tn, fn)
        self.jac += _jac(predict, target)
        self.num += 1

    def get(self):
        metrics = {
            "Recall": self.recall / self.num,
            "Precision": self.precision / self.num,
            "Specificity": self.specificity / self.num,
            "Acc": self.acc / self.num,
            "F1-score": _f1_score(self.recall / self.num, self.precision / self.num),
            "Jaccard Coefficient": self.jac / self.num
        }
        return metrics


def _jac(predict, target):
    intersection = np.count_nonzero(predict & target)
    union = np.count_nonzero(predict | target)

    jac = float(intersection) / float(union)
    return jac


def _recall(tp, fn):
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    return recall


def _specificity(tn, fp):
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    return specificity


def _precision(tp, fp):
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    return precision


def _f1_score(recall, precision):
    try:
        f1_score = 2.0 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1_score = 0.0
    return f1_score


def _accuracy(tp, fp, tn, fn):
    try:
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        acc = 0.0
    return acc
