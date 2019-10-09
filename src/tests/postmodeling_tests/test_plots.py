from triage.component.postmodeling import Model, ModelGroup, get_model, get_model_group, create_session
from triage.component.postmodeling.plots import plot_roc, plot_precision_recall_n, plot_metric_over_time
from triage.component.postmodeling.crosstabs import run_crosstabs
from tests.utils import sample_config, populate_source_data, assert_plot_figures_added
from triage.experiments import SingleThreadedExperiment
import pandas as pd
import pytest
import os

@pytest.yield_fixture(scope="module")
def session(shared_db_engine, shared_project_storage):
    """Returns an instantiated ModelEvaluator available at module scope"""
    populate_source_data(shared_db_engine)
    base_config = sample_config()
    base_config['grid_config'] = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [10],
            'criterion': ['gini'],
            'max_depth': [1],
            'max_features': ['sqrt'],
            'min_samples_split': [2],
        }
    }
    SingleThreadedExperiment(
        base_config,
        db_engine=shared_db_engine,
        project_path=shared_project_storage.project_path
    ).run()

    session = create_session(shared_db_engine)
    yield session


def test_plot_metric_over_time(session):
    model_group = get_model_group(1, session)
    with assert_plot_figures_added():
        plot_metric_over_time([model_group], metric='precision', parameter='10_pct')


def test_plot_precision_recall_n(session):
    model = get_model(1, session)
    with assert_plot_figures_added():
        plot_precision_recall_n(model)


def test_plot_ROC(session):
    model = get_model(1, session)
    with assert_plot_figures_added():
        plot_roc(model)
