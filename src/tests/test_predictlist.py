from triage.predictlist import Retrainer, predict_forward_with_existed_model, train_matrix_info_from_model_id, experiment_config_from_model_id
from triage.validation_primitives import table_should_have_data


def test_predict_forward_with_existed_model_should_write_predictions(finished_experiment):
    # given a model id and as-of-date <= today 
    # and the model id is trained and is linked to an experiment with feature and cohort config
    # generate records in triage_production.predictions
    # the # of records should equal the size of the cohort for that date
    model_id = 1
    as_of_date = '2014-01-01'
    predict_forward_with_existed_model(
            db_engine=finished_experiment.db_engine,
            project_path=finished_experiment.project_storage.project_path,
            model_id=model_id,
            as_of_date=as_of_date
    )
    table_should_have_data(
        db_engine=finished_experiment.db_engine,
        table_name="triage_production.predictions",
    )


def test_predict_forward_with_existed_model_should_be_same_shape_as_cohort(finished_experiment):
    model_id = 1
    as_of_date = '2014-01-01'
    predict_forward_with_existed_model(
            db_engine=finished_experiment.db_engine,
            project_path=finished_experiment.project_storage.project_path,
            model_id=model_id,
            as_of_date=as_of_date)

    num_records_matching_cohort = finished_experiment.db_engine.execute(
        f'''select count(*)
        from triage_production.predictions
        join triage_production.cohort_{finished_experiment.config['cohort_config']['name']} using (entity_id, as_of_date)
        '''
    ).first()[0]

    num_records = finished_experiment.db_engine.execute(
        'select count(*) from triage_production.predictions'
    ).first()[0]
    assert num_records_matching_cohort == num_records


def test_predict_forward_with_existed_model_matrix_record_is_populated(finished_experiment):
    model_id = 1
    as_of_date = '2014-01-01'
    predict_forward_with_existed_model(
            db_engine=finished_experiment.db_engine,
            project_path=finished_experiment.project_storage.project_path,
            model_id=model_id,
            as_of_date=as_of_date)

    matrix_records = list(finished_experiment.db_engine.execute(
        "select * from triage_metadata.matrices where matrix_type = 'production'"
    ))
    assert len(matrix_records) == 1


def test_experiment_config_from_model_id(finished_experiment):
    model_id = 1
    experiment_config = experiment_config_from_model_id(finished_experiment.db_engine, model_id)
    assert experiment_config == finished_experiment.config


def test_train_matrix_info_from_model_id(finished_experiment):
    model_id = 1
    (train_matrix_uuid, matrix_metadata) = train_matrix_info_from_model_id(finished_experiment.db_engine, model_id)
    assert train_matrix_uuid
    assert matrix_metadata


def test_retrain_should_write_model(finished_experiment):
    # given a model id and prediction_date 
    # and the model id is trained and is linked to an experiment with feature and cohort config
    # create matrix for retraining a model
    # generate records in production models
    # retrain_model_hash should be the same with model_hash in triage_metadata.models
    model_group_id = 1
    prediction_date = '2014-03-01'

    retrainer = Retrainer(
        db_engine=finished_experiment.db_engine,
        project_path=finished_experiment.project_storage.project_path,
        model_group_id=model_group_id,
    )
    retrain_info = retrainer.retrain(prediction_date)
    model_comment = retrain_info['retrain_model_comment']

    records = [
        row
        for row in finished_experiment.db_engine.execute(
            f"select model_hash from triage_metadata.models where model_comment = '{model_comment}'"
        )
    ]
    assert len(records) == 1
    assert retrainer.retrain_model_hash == records[0][0]

    retrainer.predict(prediction_date)
    
    table_should_have_data(
        db_engine=finished_experiment.db_engine,
        table_name="triage_production.predictions",
    )
    
    matrix_records = list(finished_experiment.db_engine.execute(
        f"select * from triage_metadata.matrices where matrix_uuid = '{retrainer.predict_matrix_uuid}'"
    ))
    assert len(matrix_records) == 1
