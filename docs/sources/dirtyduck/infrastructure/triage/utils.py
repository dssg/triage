# coding: utf-8

import os
import json
from pprint import pprint

import sqlalchemy
from sqlalchemy import create_engine

from io import StringIO
from functools import reduce
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
sns.set_style("ticks")

import pydotplus

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.externals import joblib

import triage.component.timechop as timechop
from triage.util.conf import convert_str_to_relativedelta

from triage.component.audition.distance_from_best import DistanceFromBestTable,\
                                                         BestDistancePlotter
from triage.component.audition.pre_audition import PreAudition

from triage.component.audition import Auditioner

from triage.component.audition.rules_maker import SimpleRuleMaker, \
                                                  RandomGroupRuleMaker, \
                                                  create_selection_grid


FIG_SIZE = (32,16)
TRIAGE_DB_URL = os.environ.get("TRIAGE_DB_URL")
TRIAGE_OUTPUT_PATH = os.environ.get("TRIAGE_OUTPUT_PATH")

def show_timechop(chopper, show_as_of_times=True, show_boundaries=True, file_name=None):

    plt.close('all')

    chops = chopper.chop_time()

    chops.reverse()

    fig, ax = plt.subplots(len(chops), sharex=True, sharey=True, figsize=FIG_SIZE)


    for idx, chop in enumerate(chops):
        train_as_of_times = chop['train_matrix']['as_of_times']
        test_as_of_times = chop['test_matrices'][0]['as_of_times']

        max_training_history = chop['train_matrix']['max_training_history']
        test_label_timespan = chop['test_matrices'][0]['test_label_timespan']
        training_label_timespan = chop['train_matrix']['training_label_timespan']

        color_rgb = np.random.random(3)

        if(show_as_of_times):
            # Train matrix (as_of_times)
            ax[idx].hlines(
              [x for x in range(len(train_as_of_times))],
              [x.date() for x in train_as_of_times],
              [x.date() + convert_str_to_relativedelta(training_label_timespan) for x in train_as_of_times],
              linewidth=3, color=color_rgb,label=f"train_{idx}"
            )

            # Test matrix
            ax[idx].hlines(
              [x for x in range(len(test_as_of_times))],
              [x.date() for x in test_as_of_times],
              [x.date() + convert_str_to_relativedelta(test_label_timespan) for x in test_as_of_times],
              linewidth=3, color=color_rgb,
              label=f"test_{idx}"
            )


        if(show_boundaries):
            # Limits: train
            ax[idx].axvspan(chop['train_matrix']['first_as_of_time'],
                            chop['train_matrix']['last_as_of_time'],
                            color=color_rgb,
                            alpha=0.3
            )


            ax[idx].axvline(chop['train_matrix']['matrix_info_end_time'], color='k', linestyle='--')


            # Limits: test
            ax[idx].axvspan(chop['test_matrices'][0]['first_as_of_time'],
                            chop['test_matrices'][0]['last_as_of_time'],
                            color=color_rgb,
                            alpha=0.3
            )

            ax[idx].axvline(chop['feature_start_time'], color='k', linestyle='--', alpha=0.2)
            ax[idx].axvline(chop['feature_end_time'], color='k', linestyle='--',  alpha=0.2)
            ax[idx].axvline(chop['label_start_time'] ,color='k', linestyle='--', alpha=0.2)
            ax[idx].axvline(chop['label_end_time'] ,color='k', linestyle='--',  alpha=0.2)

            ax[idx].axvline(chop['test_matrices'][0]['matrix_info_end_time'],color='k', linestyle='--')

        ax[idx].yaxis.set_major_locator(plt.NullLocator())
        ax[idx].yaxis.set_label_position("right")
        ax[idx].set_ylabel(f"Block {idx}", rotation='horizontal', labelpad=30)

        ax[idx].xaxis.set_major_formatter(md.DateFormatter('%Y'))
        ax[idx].xaxis.set_major_locator(md.YearLocator())
        ax[idx].xaxis.set_minor_locator(md.MonthLocator())

    ax[0].set_title('Timechop: Temporal cross-validation blocks')
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    file_name = os.path.join(TRIAGE_OUTPUT_PATH, "images", file_name)
    fig.savefig(file_name)

    plt.show()

    return file_name


def show_features_queries(st):

    for sql_list in st.get_selects().values():
        for sql in sql_list:
            print(str(sql))

    print(str(st.get_create()))


def get_model_hashes(model_id):
    db = create_engine(TRIAGE_DB_URL)

    rows = db.execute(
        f"""
        select distinct on (model_hash, train_matrix_uuid, matrix_uuid)
        model_hash, train_matrix_uuid as train_hash, matrix_uuid as test_hash
        from model_metadata.models
        inner join test_results.predictions using(model_id)
        where model_id = {model_id};
       """)

    for row in rows:
        model_hash, train_hash, test_hash = row.model_hash, row.train_hash, row.test_hash

    return model_hash, train_hash, test_hash

def show_model(model_id):
    model_hash, train_hash, _ = get_model_hashes(model_id)

    clf = joblib.load(os.path.join(TRIAGE_OUTPUT_PATH, "trained_models", model_hash))

    X = pd.read_csv(os.path.join(TRIAGE_OUTPUT_PATH, "matrices", f"{train_hash}.csv"), nrows = 1)

    ## The first two columns are ALWAYS entity_id, as_of_date and the last one in the label
    X.drop(columns=X.columns[[0,1,-1]], axis = 1, inplace=True)

    trees = []
    file_names = []

    if isinstance(clf, RandomForestClassifier):
        # We have a forest, we will pick 5 at random
        trees.extend(np.random.choice(clf.estimators_, size=5 , replace=False))
        max_depth = 5
        print("""
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        IMPORTANT: The decisions trees are being cropped to a maximum depth of 5.
        If your tree is bigger, remember that you aren't viewing the FULL tree.

        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """)

    elif isinstance(clf, DecisionTreeClassifier):
        trees.append(clf)
        max_depth = None
    else:
        trees = None
        file_names = None
        print("You selected a model that isn't a Decision Tree. I can not plot that. Sorry")

    for i, dtree in enumerate(trees):
        print(f"Plotting tree number {i}")
        dot_data = StringIO()
        export_graphviz(dtree,out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names = X.columns,
                        class_names=True,
                        max_depth=max_depth)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        file_name = os.path.join(TRIAGE_OUTPUT_PATH, "images", f"model_{model_id}_tree_{i}.svg")
        graph.write_svg(file_name)
        file_names.append(file_name)

    return file_names

def audit_experiment(experiment_hash, metric, k, rules):
    db = create_engine(TRIAGE_DB_URL)

    pre_audition = PreAudition(db)

    model_groups = pre_audition.get_model_groups_from_experiment(experiment_hash)
    end_times = pre_audition.get_train_end_times(query=f"""
                             select distinct train_end_time,  experiment_hash
                             from model_metadata.models
                             where experiment_hash = '{experiment_hash}'
                """)

    auditioner = Auditioner(
        db_engine = db,
        model_group_ids = model_groups,
        train_end_times = end_times,
        initial_metric_filters = [{
            'metric': metric,
            'parameter': k,
            'max_from_best': 0.1,
            'threshold_value': 0.0
        }],
        models_table = 'models',
        distance_table = 'postmodel.model_distances'
    )

    auditioner.register_selection_rule_grid(rules)

    print("""
          ++++++++++++++++++++++++++++++++++++++++++++++++++++
          +                                                  +
          +          Results of the simulation               +
          +                                                  +
          ++++++++++++++++++++++++++++++++++++++++++++++++++++
    """)
    pprint(auditioner.selection_rule_model_group_ids)

    print("""
          ++++++++++++++++++++++++++++++++++++++++++++++++++++
          +                                                  +
          +          Average regret per rule                 +
          +                                                  +
          ++++++++++++++++++++++++++++++++++++++++++++++++++++
    """)

    pprint(auditioner.average_regret_for_rules)

# def compare_models_lists(models):
#     db = create_engine(TRIAGE_DB_URL)

#     list_results=get_list_comparision(conn=conn,
#                                   model_ids=model_ids_100abs,
#                                   test_time=test_time,
#                                   top_k=100)

#     list_results = pd.DataFrame(list_results)

#     list_results[['model_0_id','model_1_id', 'jaccard', 'overall_rank_corr']].sort_values(['jaccard'], ascending=[False]).head(10)


# def compare_models_features(models):
#     db = create_engine(TRIAGE_DB_URL)

#     list_results=get_feature_list_comparision(
#         conn=conn,
#         model_ids=model_ids_100abs,
#         buckets=None,
#         top_k=50)

#     result_df = pd.DataFrame(list_results)

#     result_df[['model_0_id','model_1_id', 'jaccard', 'overall_rank_corr']].sort_values(['jaccard'], ascending=[False]).head(10)

# def score_distributions(model_group):
#     db = create_engine(TRIAGE_DB_URL)

#     for model in models:
#         model_id = model[1]
#         test_time = model[-1]
#         print(f"Model: {model_id}")
#         df_scores=get_score_distribution(conn=db,
#                                          model_id=model_id,
#                                          test_time=test_time,
#                                          plot=True,
#                                          filename=f"../images/sc_119_{model_id}_{test_time}.png")

# def probability_curves(model_group):
#     for model in models:
#         model_id = model[1]
#         test_time = model[-1]
#         print(f"Model: {model_id}")
#         df_scores=get_probability_by_decile(conn=db,
#                                             model_id=model_id,
#                                             test_time=test_time,
#                                             plot=True,
#                                             filename=f"../images/pc_119_{model_id}_{test_time}.png")
