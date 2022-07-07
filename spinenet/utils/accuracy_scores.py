import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score


def balanced_accuracy(y_label, y_pred):
    # Move to cpu
    y_pred = np.array(y_pred, dtype=float)
    y_label = np.array(y_label, dtype=float)
    # Delete ignore_index = -100
    y_pred = np.delete(y_pred, np.where(y_label == -100))
    y_label = np.delete(y_label, np.where(y_label == -100))
    return balanced_accuracy_score(y_label, y_pred)


# binarized score before determining balanced accuracy e.g. for normal vs abnormal accuracy
def binarized_balanced_accuracy(y_label, y_pred):
    y_pred = np.array(y_pred, dtype=float)
    y_label = np.array(y_label, dtype=float)
    y_label[np.where(y_label > 1)] = 1
    y_pred[np.where(y_pred > 1)] = 1
    # Delete ignore_index = -100
    y_pred = np.delete(y_pred, np.where(y_label == -100))
    y_label = np.delete(y_label, np.where(y_label == -100))
    return balanced_accuracy_score(y_label, y_pred)
