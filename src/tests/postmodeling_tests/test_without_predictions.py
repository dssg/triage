from triage.component.postmodeling.contrast.model_group_evaluator import ModelGroupEvaluator
from triage.component.postmodeling.contrast.model_evaluator import ModelEvaluator
import pytest


@pytest.fixture(scope="module")
def model_group_evaluator(finished_experiment_without_predictions):
    return ModelGroupEvaluator((1, 1), finished_experiment_without_predictions.db_engine)


@pytest.fixture(scope="module")
def model_evaluator(finished_experiment_without_predictions):
    return ModelEvaluator(1, 1, finished_experiment_without_predictions.db_engine)


def test_ModelGroupEvaluator_metadata(model_group_evaluator):
    assert all(value for metadata_row in model_group_evaluator.metadata for key, value in metadata_row.items() )


def test_ModelGroupEvaluator_predictions(model_group_evaluator):
    with pytest.raises(RuntimeError):
        model_group_evaluator.predictions


def test_ModelEvaluator_metadata(model_evaluator):
    assert all(value for key, value in model_evaluator.metadata.items())


def test_ModelEvaluator_predictions(model_evaluator):
    with pytest.raises(RuntimeError):
        model_evaluator.predictions
