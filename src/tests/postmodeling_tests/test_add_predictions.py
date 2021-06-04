import pytest

from triage.component.postmodeling.utils.predictions_selected_model_groups import generate_predictions

MODEL_IDS_QUERY = """
    SELECT
        model_id
    FROM triage_metadata.models
    WHERE model_group_id={model_group_id}
"""

PREDICTIONS_QUERY = """
    SELECT 
        model_id,
        entity_id,
        as_of_date,
        score
    FROM test_results.predictions
    WHERE model_id={model_id}
"""


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
        model_ids = db_engine.execute(MODEL_IDS_QUERY.format(model_group_id))

        # Asserting that all model_ids in the groups have the predictions written to the DB
        for model_id in model_ids:
            predictions = db_engine.execute(PREDICTIONS_QUERY.format(model_id))

            assert len(predictions) > 0