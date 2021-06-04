import pytest

from triage.component.postmodeling.contrast.model_group_evaluator import ModelGroupEvaluator
from triage.component.postmodeling.contrast.model_evaluator import ModelEvaluator
from triage.component.postmodeling.utils.predictions_selected_model_groups import generate_predictions

def test_add_predictions(finished_experiment_without_predictions):
    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [1]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    generate_predictions(
        db_engine=db_engine,
        model_groups=model_groups,
        project_path=project_path
    )

    for model_group_id in model_groups:
        model_group_evaluator = ModelGroupEvaluator(model_group_id, db_engine)
        model_ids = model_group_evaluator.model_id

        # Asserting that all model_ids in the groups have the predictions written to the DB
        for model_id in model_ids:
            model_evaluator = ModelEvaluator(model_group_id, model_id, db_engine)
            try:
                model_evaluator.predictions
            except RuntimeError as e:
                pytest.fail('Predictions not saved RuntimeError -- {}'.format(e))
