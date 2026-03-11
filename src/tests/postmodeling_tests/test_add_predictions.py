import pytest

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from tests.conftest import finished_experiment
from triage.component.postmodeling.add_predictions import add_predictions
from triage.database_reflection import table_has_data

# MODEL_IDS_QUERY = """
#     SELECT
#         model_id
#     FROM triage_metadata.models
#     WHERE model_group_id={model_group_id}
# """

# MODELS_IN_PREDICTIONS_QUERY = """
#     SELECT 
#         distinct model_id
#     FROM test_results.predictions
# """


# def test_populate_predictions_table(finished_experiment_without_predictions):
#     """assert that generate_predictions populate the predictions table"""

#     db_engine = finished_experiment_without_predictions.db_engine
#     model_groups = [1]
#     project_path = finished_experiment_without_predictions.project_storage.project_path

#     add_predictions(
#         db_engine=db_engine,
#         model_groups=model_groups,
#         project_path=project_path
#     )

#     assert table_has_data('test_results.predictions', db_engine)


# def test_add_predictions_all_models(finished_experiment_without_predictions):
#     """assert that generate_predictions write predictions of all models in the model group"""

#     db_engine = finished_experiment_without_predictions.db_engine
#     model_groups = [1]
#     project_path = finished_experiment_without_predictions.project_storage.project_path

#     add_predictions(
#         db_engine=db_engine,
#         model_groups=model_groups,
#         project_path=project_path
#     )

#     # Model ids belonging to the model group 1
#     with db_engine.connect() as conn:
#         model_ids = conn.execute(text(MODEL_IDS_QUERY.format(model_group_id=1))).fetchall()
#     model_ids = {x[0] for x in model_ids}

#     # model ids present in the predictions table
#     with db_engine.connect() as conn:
#         model_ids_predictions = conn.execute(text(MODELS_IN_PREDICTIONS_QUERY)).fetchall()
#     model_ids_predictions = {x[0] for x in model_ids_predictions}
        
#     assert model_ids == model_ids_predictions

def test_add_predictions_from_experiment_hash(finished_experiment_without_predictions):
    """Assert that predictions from all model group ids associated with the experiment hash are added"""

    project_path = finished_experiment_without_predictions.project_storage.project_path
    experiment_hash = finished_experiment_without_predictions.experiment_hash
        
    # verify that we don't have predictions yet
    assert not table_has_data('test_results.predictions', finished_experiment_without_predictions.db_engine)

    query = """
            select distinct model_group_id
            from triage_metadata.experiment_models a 
            join triage_metadata.models b
             using (model_hash)
            where experiment_hash = :experiment_hash
        """
    
    with finished_experiment_without_predictions.db_engine.connect() as conn:
        result = conn.execute(
            text(query),
            {'experiment_hash': experiment_hash}
        ).fetchall()
    model_group_ids = [row[0] for row in result] 
    # we have 4 model groups associated with this experiment
    assert len(result) == 4

    add_predictions(
        db_engine=finished_experiment_without_predictions.db_engine,
        model_groups=model_group_ids,
        project_path=project_path,
        experiment_hashes=[experiment_hash]
    )

    assert table_has_data('test_results.predictions', finished_experiment_without_predictions.db_engine)


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


def test_add_predictions_from_model_group_id(finished_experiment_without_predictions):
    """Assert that predictions are added for a specific model group id"""
    project_path = finished_experiment_without_predictions.project_storage.project_path
    model_groups = [1]  

    # first check that we don't have predictions for that model group id 
    with finished_experiment_without_predictions.db_engine.connect() as conn:
        result = conn.execute(
            text("""
                select distinct model_id
                from triage_metadata.models 
                join test_results.predictions 
                 using(model_id)
                where model_group_id = :model_group_id
            """),
            {'model_group_id': 1}
        ).fetchall()
    model_ids_w_predictions = [row[0] for row in result]
    assert len(model_ids_w_predictions) == 0    

    add_predictions(
        db_engine=finished_experiment_without_predictions.db_engine,
        model_groups=model_groups,
        project_path=project_path
    )
    # now check that we have predictions for that model group id
    with finished_experiment_without_predictions.db_engine.connect() as conn:
        result = conn.execute(
            text("""
                select distinct model_id
                from triage_metadata.models 
                join test_results.predictions 
                 using(model_id)
                where model_group_id = :model_group_id
            """),
            {'model_group_id': 1}
        ).fetchall()
    model_ids_w_predictions = [row[0] for row in result]
    assert len(model_ids_w_predictions) == 2
    

def test_add_predictions_predictions_already_exist(finished_experiment):
    """
    Assert that no predictions are added if they already exist and replace is False 
    """
    db_engine = finished_experiment.db_engine
    project_path = finished_experiment.project_storage.project_path
    model_groups = [1]

    # verify that we have predictions 
    query = """
        select distinct model_id
        from triage_metadata.models 
        join test_results.predictions 
         using(model_id)
        where model_group_id = :model_group_id
    """
    with db_engine.connect() as conn:
        result = conn.execute(
            text(query),
            {'model_group_id': 1}
        ).fetchall()
    model_ids_w_predictions = [row[0] for row in result]
    # there should be predictions for 2 models in model group 1
    assert len(model_ids_w_predictions) == 2

    add_predictions(
        db_engine=db_engine,
        model_groups=model_groups,
        project_path=project_path,
        replace=False
    )

    with db_engine.connect() as conn:
        result = conn.execute(
            text(query),
            {'model_group_id': 1}
        ).fetchall()
    model_ids_w_predictions = [row[0] for row in result]
    # there should be predictions for 2 models in model group 1
    assert len(model_ids_w_predictions) == 2


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

    with db_engine.connect() as conn:
        end_times = conn.execute(text(q)).fetchall()[0]
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
