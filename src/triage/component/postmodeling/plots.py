# coding: utf-8


import itertools

import math
import matplotlib
#matplotlib.use('Agg')  ## Needed since we could use this in non-interactive mode
                       ## see https://matplotlib.org/faq/howto_faq.html

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper", rc={"font.size": 8, "axes.labelsize": 5})

import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score
from sklearn.tree import export_graphviz
import pydotplus

import numpy as np
import pandas as pd

from triage.component.postmodeling import get_predictions, get_model, get_model_group, get_evaluations


def plot_roc(model, output_type='show', **kwargs):
    predictions = get_predictions(model)
    labels = predictions.label_value
    scores = predictions.score
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(kwargs.get('title', ''))
    plt.legend(loc="lower right")

    if (output_type == 'save'):
        filename = kwargs('filename', f"{model}")
        extension = kwargs('extension', 'png')
        plt.savefig(f"{filename}.{extension}")
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def precision_n(y_true, y_scores):
    precisions = []
    for i in range(101):
        prec = precision_at_k(y_true, y_scores, i)
        precisions.append(prec)
    return precisions


def recall_n(y_true, y_scores):
    recalls = []
    for i in range(101):
        rec = recall_at_k(y_true, y_scores, i)
        recalls.append(rec)
    return recalls


def plot_precision_recall_n(model, output_type='show', **kwargs):
    predictions = get_predictions(model)
    labels = predictions.label_value
    scores = predictions.score

    y_axis = np.arange(0, 101, 1)

    prec_k = precision_n(labels, scores)
    rec_k = recall_n(labels, scores)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(y_axis, prec_k, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(y_axis, rec_k, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,1])
    ax1.set_xlim([0,100])
    ax2.set_xlim([0,100])

    plt.title(kwargs.get('title', ''))
    if (output_type == 'save'):
        filename = kwargs('filename', f"{model}")
        extension = kwargs('extension', 'png')
        plt.savefig(f"{filename}.{extension}")
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


def get_subsets(l):
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]
