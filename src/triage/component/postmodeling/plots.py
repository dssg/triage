# coding: utf-8

import itertools
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

import math

import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use('Agg')  ## Needed since we could use this in non-interactive mode
                       ## see https://matplotlib.org/faq/howto_faq.html

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import seaborn as sns
sns.set(style="darkgrid")


import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score
from sklearn.tree import export_graphviz

import pydotplus


from triage.component.postmodeling import get_predictions, get_model, get_model_group, get_evaluations
from triage.component.results_schema import ModelGroup, Model

def _store(fig, **kwargs):
    output_type=kwargs.get('output_type', 'show')

    if (output_type == 'save'):
        filename = kwargs.get('filename', f"{model}")
        extension = kwargs.get('extension', 'png')
        transparent = kwargs.get('transparent', False)
        dpi = kwargs.get('dpi', 80)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(f"{filename}.{extension}", transparent=transparent, dpi=dpi, bbox_inches=bbox_inches)


def plot_roc(model: Model, **kwargs):
    fig, ax = plt.subplots()

    predictions = get_predictions(model)

    label_value_counts = predictions.label_value.value_counts(dropna=False)
    logging.warning(f"Label Value Counts: \n{label_value_counts}")
    if np.nan in label_value_counts:
        logging.warning("There are NaNs in label_value. Ignore NaNs when calculating ROC curve.")
    predictions = predictions.dropna(subset=['label_value'])

    labels = predictions.label_value
    scores = predictions.score

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fmt = kwargs.get('fmt', 'b')

    # Random performance
    ax.plot([0, 1], [0, 1], 'k--')

    # Actual model
    ax.plot(fpr, tpr, fmt, label='ROC curve (area = %0.2f)' % roc_auc)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.set_title(kwargs.get('title', f"(Model {model.model_id} @ {model.train_end_time.strftime('%Y-%m-%d')})"))

    ax.legend(loc="lower right")

    fig.suptitle(kwargs.get('suptitle', f"Model Group {model.model_group_rel.model_group_id}"))

    _store(fig, **kwargs)

    return fig, ax


def plot_precision_recall_n(model: Model, **kwargs):
    fig, ax1 = plt.subplots()

    predictions = get_predictions(model)

    label_value_counts = predictions.label_value.value_counts(dropna=False)
    logging.warning(f"Label Value Counts: \n{label_value_counts}")
    if np.nan in label_value_counts:
        logging.warning("There are NaNs in label_value. Ignore NaNs when calculating ROC curve.")
    predictions = predictions.dropna(subset=['label_value'])

    labels = predictions.label_value
    scores = predictions.score

    y_axis = np.arange(0, 101, 1)

    prec_k = precision_n(labels, scores)
    rec_k = recall_n(labels, scores)

    prec_color = kwargs.get('prec_fmt', 'tab:blue')
    rec_color = kwargs.get('rec_fmt', 'tab:red')

    ax1.plot(y_axis, prec_k, prec_color)
    ax1.set_xlabel('Percent of population')
    ax1.set_ylabel('Precision', color=prec_color)
    ax1.tick_params(axis='y', labelcolor=prec_color)

    ax1.set_ylim([0,1])
    ax1.set_xlim([0,100])

    ax1.set_title(kwargs.get('title', f"(Model {model.model_id} @ {model.train_end_time.strftime('%Y-%m-%d')})"))

    ax2 = ax1.twinx()
    ax2.plot(y_axis, rec_k, rec_color)
    ax2.set_ylim([0,1])
    ax2.set_ylabel('Recall', color=rec_color)
    ax2.tick_params(axis='y', labelcolor=rec_color)


    fig.suptitle(kwargs.get('suptitle', f"Model Group {model.model_group_rel.model_group_id}"))

    _store(fig, **kwargs)

    return fig, ax1, ax2


def plot_metric_over_time(model_groups: List[ModelGroup], metric: str, parameter: str, **kwargs):

    def get_model_group_evaluations(model_group):
        evaluations = pd.concat(get_evaluations(model).query(f"metric == '{metric}@' and parameter == '{parameter}'") for model in model_group)
        evaluations = evaluations.sort_values('evaluation_start_time', ascending=True)
        evaluations = evaluations[['evaluation_start_time', 'stochastic_value']]
        evaluations['model_type'] = model_group.model_type.split('.')[-1]
        return evaluations

    metric = metric.replace('@','')

    fig, ax = plt.subplots()

    evaluations = pd.concat(get_model_group_evaluations(model_group) for model_group in model_groups)

    for name, mg in evaluations.groupby(by='model_group_id'):
        model_type = mg.model_type.unique()[0]
        ax.plot_date(x=mg.evaluation_start_time, y=mg.stochastic_value,  xdate=True, linestyle='-', label=f"Model group {name} ({model_type})")
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    ax.set_xlabel('')
    ax.set_ylabel(f"{metric.capitalize()} at {parameter.replace('_', ' ').replace('pct', '%')}")

    ax.set_ylim([0, 1])

    ax.legend(loc='best',
              borderaxespad=0.,
              title='Model Group')

    ax.set_title(kwargs.get('title', f"Model Groups: performance over time"))

    ax.grid(which='minor', linewidth='0.5', alpha=0.8)
    ax.grid(which='major', linewidth='3.0', alpha=0.3)

    ax.tick_params(which='major', # Options for both major and minor ticks
                   top='off', # turn off top ticks
                   left='off', # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='on') # turn off bottom ticks

    _store(fig, **kwargs)

    return fig, ax


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
