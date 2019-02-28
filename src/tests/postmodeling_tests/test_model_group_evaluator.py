from triage.component.postmodeling.contrast.model_group_evaluator import ModelGroupEvaluator
import pandas as pd
import pytest
from tests.utils import assert_plot_figures_added


@pytest.fixture(scope="module")
def model_group_evaluator(finished_experiment):
    return ModelGroupEvaluator((1,1), finished_experiment.db_engine)


def test_ModelGroupEvaluator_metadata(model_group_evaluator):
    assert isinstance(model_group_evaluator.metadata, list)
    assert len(model_group_evaluator.metadata) == 8 # 8 model groups expected from basic experiment
    for row in model_group_evaluator.metadata:
        assert isinstance(row, dict)
        

def test_ModelGroupEvaluator_model_type(model_group_evaluator):
    assert model_group_evaluator.model_type[0] == 'sklearn.tree.DecisionTreeClassifier'


def test_ModelGroupEvaluator_predictions(model_group_evaluator):
    assert isinstance(model_group_evaluator.predictions, pd.DataFrame)


def test_ModelGroupEvaluator_feature_importances(model_group_evaluator):
    assert isinstance(model_group_evaluator.feature_importances, pd.DataFrame)


def test_ModelGroupEvaluator_metrics(model_group_evaluator):
    assert isinstance(model_group_evaluator.metrics, pd.DataFrame)


def test_ModelGroupEvaluator_feature_groups(model_group_evaluator):
    assert isinstance(model_group_evaluator.feature_groups, pd.DataFrame)


def test_ModelGroupEvaluator_same_time_models(model_group_evaluator):
    assert isinstance(model_group_evaluator.same_time_models, pd.DataFrame)


def test_ModelGroupEvaluator_plot_prec_across_time(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_prec_across_time()


def test_ModelGroupEvaluator_feature_loi_loo(model_group_evaluator):
    with pytest.raises(IndexError):
        model_group_evaluator.feature_loi_loo()


def test_ModelGroupEvaluator_plot_ranked_correlation_preds(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_ranked_correlation_preds(param_type='rank_abs', param=10, top_n_features=10)


def test_ModelGroupEvaluator_plot_ranked_correlation_features(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_ranked_correlation_features(param_type='rank_abs', param=10, top_n_features=10)

def test_ModelGroupEvaluator_plot_jaccard_preds(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_jaccard_preds(param_type='rank_abs', param=10)


def test_ModelGroupEvaluator_plot_jaccard_features(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_jaccard_features()


def test_ModelGroupEvaluator_plot_preds_comparison(model_group_evaluator):
    with assert_plot_figures_added():
        model_group_evaluator.plot_preds_comparison(param_type='rank_abs', param=10)
