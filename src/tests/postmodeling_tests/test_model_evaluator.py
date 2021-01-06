from triage.component.postmodeling.contrast.model_evaluator import ModelEvaluator
from triage.component.postmodeling.crosstabs import run_crosstabs
from tests.utils import sample_config, populate_source_data, assert_plot_figures_added
from triage.experiments import SingleThreadedExperiment
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def model_evaluator(shared_db_engine, shared_project_storage):
    """Returns an instantiated ModelEvaluator available at module scope"""
    populate_source_data(shared_db_engine)
    base_config = sample_config()
    # We need to have an ensemble model to test ModelEvaluator correctly
    # so we can't use the finished_experiment fixture"""
    base_config["grid_config"] = {
        "sklearn.ensemble.ExtraTreesClassifier": {
            "n_estimators": [10],
            "criterion": ["gini"],
            "max_depth": [1],
            "max_features": ["sqrt"],
            "min_samples_split": [2],
        }
    }
    SingleThreadedExperiment(
        base_config,
        db_engine=shared_db_engine,
        project_path=shared_project_storage.project_path,
    ).run()
    return ModelEvaluator(1, 1, shared_db_engine)


def test_ModelEvaluator_model_type(model_evaluator):
    assert model_evaluator.model_type == "sklearn.ensemble.ExtraTreesClassifier"


def test_ModelEvaluator_predictions(model_evaluator):
    assert isinstance(model_evaluator.predictions, pd.DataFrame)


def test_ModelEvaluator_feature_importances(model_evaluator):
    assert isinstance(model_evaluator.feature_importances(), pd.DataFrame)


def test_ModelEvaluator_feature_group_importances(model_evaluator):
    assert isinstance(model_evaluator.feature_group_importances(), pd.DataFrame)


def test_ModelEvaluator_test_metrics(model_evaluator):
    assert isinstance(model_evaluator.test_metrics, pd.DataFrame)


def test_ModelEvaluator_train_metrics(model_evaluator):
    assert isinstance(model_evaluator.train_metrics, pd.DataFrame)


def test_ModelEvaluator_crosstabs(model_evaluator, crosstabs_config):
    run_crosstabs(model_evaluator.engine, crosstabs_config)
    assert isinstance(model_evaluator.crosstabs, pd.DataFrame)


def test_ModelEvaluator_preds_matrix(model_evaluator, shared_project_storage):
    assert isinstance(
        model_evaluator.preds_matrix(shared_project_storage.project_path), pd.DataFrame
    )


def test_ModelEvaluator_plot_score_distribution(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_score_distribution()


def test_ModelEvaluator_plot_score_label_distributions(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_score_label_distributions()


def test_ModelEvaluator_plot_score_distribution_thresh(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_score_distribution_thresh(param_type="rank_abs", param=10)


def test_ModelEvaluator_plot_feature_importances(
    model_evaluator, shared_project_storage
):
    with assert_plot_figures_added():
        model_evaluator.plot_feature_importances(shared_project_storage.project_path)


def test_ModelEvaluator_plot_feature_importances_std_err(
    model_evaluator, shared_project_storage
):
    with assert_plot_figures_added():
        model_evaluator.plot_feature_importances_std_err(
            shared_project_storage.project_path
        )


def test_ModelEvaluator_plot_precision_recall_n(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_precision_recall_n()


def test_ModelEvaluator_plot_recall_fpr_n(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_recall_fpr_n()


def test_ModelEvaluator_plot_ROC(model_evaluator):
    with assert_plot_figures_added():
        model_evaluator.plot_ROC()


def test_ModelEvaluator_compute_AUC(model_evaluator):
    assert isinstance(model_evaluator.compute_AUC(), tuple)


def test_ModelEvaluator_cluster_correlation_sparsity(
    model_evaluator, shared_project_storage
):
    with assert_plot_figures_added():
        model_evaluator.cluster_correlation_sparsity(
            shared_project_storage.project_path
        )


def test_ModelEvaluator_cluster_correlation_features(
    model_evaluator, shared_project_storage
):
    with assert_plot_figures_added():
        model_evaluator.cluster_correlation_features(
            shared_project_storage.project_path
        )


def test_ModelEvaluator_plot_feature_group_aggregate_importances(
    model_evaluator, shared_project_storage
):
    with assert_plot_figures_added():
        model_evaluator.plot_feature_group_aggregate_importances(
            path=shared_project_storage.project_path
        )
