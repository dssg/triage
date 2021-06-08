import pytest

from triage.component.postmodeling.utils.predictions_selected_model_groups import generate_predictions
from triage.component.architect.database_reflection import table_has_data


MODEL_IDS_QUERY = """
    SELECT
        model_id
    FROM triage_metadata.models
    WHERE model_group_id={model_group_id}
"""

MODELS_IN_PREDICTIONS_QUERY = """
    SELECT 
        distinct model_id
    FROM test_results.predictions
"""


def test_populate_predictions_table(finished_experiment_without_predictions):
    """assert that generate_predictions populate the predictions table"""

    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [1]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    generate_predictions(
        db_engine=db_engine,
        model_groups=model_groups,
        project_path=project_path
    )

    assert table_has_data('test_results.predictions', db_engine)


def test_add_predictions_all_models(finished_experiment_without_predictions):
    """assert that generate_predictions write predictions of all models in the model group"""

    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [1]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    generate_predictions(
        db_engine=db_engine,
        model_groups=model_groups,
        project_path=project_path
    )

    # Model ids belonging to the model group 1
    model_ids = db_engine.execute(MODEL_IDS_QUERY.format(model_group_id=1)).fetchall()
    model_ids = {x[0] for x in model_ids}

    # model ids present in the predictions table
    model_ids_predictions = db_engine.execute(MODELS_IN_PREDICTIONS_QUERY).fetchall()
    model_ids_predictions = {x[0] for x in model_ids_predictions}
        
    assert model_ids == model_ids_predictions
