import pytest

from triage.component.postmodeling.utils.add_predictions import add_predictions
from triage.database_reflection import table_has_data


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

    add_predictions(
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

    add_predictions(
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


def test_models_invalid_model_group(finished_experiment_without_predictions):
    """Test whether the module produces the error for invalid model groups"""

    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [325]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    with pytest.raises(ValueError):
        add_predictions(
            db_engine=db_engine,
            model_groups=model_groups,
            project_path=project_path
        )


def test_models_not_found_invalid_time_range(finished_experiment_without_predictions):
    """Test whether the module errors when an invalid time range is provided"""
    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [1]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    q = """
        SELECT 
            to_char(min(train_end_time) - '1month'::interval, 'YYYY-MM-DD') as lower_lim,
            to_char(max(train_end_time) + '1month'::interval, 'YYYY-MM-DD') as upper_lim
        FROM triage_metadata.models
        where model_group_id = 1

    """

    end_times = db_engine.execute(q).fetchall()[0]
    lower_lim = end_times[0]
    upper_lim = end_times[1]

    with pytest.raises(ValueError):
        add_predictions(
            db_engine=db_engine,
            model_groups=model_groups,
            project_path=project_path,
            train_end_times_range={
                'range_start_date': upper_lim,
            }
        )

    with pytest.raises(ValueError):
        add_predictions(
            db_engine=db_engine,
            model_groups=model_groups,
            project_path=project_path,
            train_end_times_range={
                'range_end_date': lower_lim
            }
        )

    with pytest.raises(ValueError):
        add_predictions(
            db_engine=db_engine,
            model_groups=model_groups,
            project_path=project_path,
            train_end_times_range={
                'range_start_date': upper_lim,
                'range_end_date': upper_lim
            }
        )


def test_invalid_experiment_hash(finished_experiment_without_predictions):
    """Test whether an invalid experiment hash throws a ValueError"""

    db_engine = finished_experiment_without_predictions.db_engine
    model_groups = [1]
    project_path = finished_experiment_without_predictions.project_storage.project_path

    experiment_hash = finished_experiment_without_predictions.experiment_hash
    experiment_hash = experiment_hash + 'x'

    with pytest.raises(ValueError):
        add_predictions(
            db_engine=db_engine,
            model_groups=model_groups,
            project_path=project_path,
            experiment_hashes=[experiment_hash]
        )
