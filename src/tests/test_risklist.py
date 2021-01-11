from triage.risklist import generate_risk_list, train_matrix_info_from_model_id, experiment_config_from_model_id
from triage.validation_primitives import table_should_have_data


def test_risklist_should_write_predictions(finished_experiment):
    # given a model id and as-of-date <= today
    # and the model id is trained and is linked to an experiment with feature and cohort config
    # generate records in listpredictions
    # the # of records should equal the size of the cohort for that date
    model_id = 1
    as_of_date = '2013-01-01'
    generate_risk_list(
            db_engine=finished_experiment.db_engine,
            project_storage=finished_experiment.project_storage,
            model_id=model_id,
            as_of_date=as_of_date)
    table_should_have_data(
        db_engine=finished_experiment.db_engine,
        table_name="production.list_predictions",
    )


def test_risklist_should_be_same_shape_as_cohort(finished_experiment):
    model_id = 1
    as_of_date = '2013-01-01'
    generate_risk_list(
            db_engine=finished_experiment.db_engine,
            project_storage=finished_experiment.project_storage,
            model_id=model_id,
            as_of_date=as_of_date)

    num_records_matching_cohort = finished_experiment.db_engine.execute(
        f'''select count(*)
        from production.list_predictions
        join production.cohort_{finished_experiment.config['cohort_config']['name']} using (entity_id, as_of_date)
        '''
    ).first()[0]

    num_records = finished_experiment.db_engine.execute(
        'select count(*) from production.list_predictions'
    ).first()[0]
    assert num_records_matching_cohort == num_records


def test_risklist_matrix_record_is_populated(finished_experiment):
    model_id = 1
    as_of_date = '2013-01-01'
    generate_risk_list(
            db_engine=finished_experiment.db_engine,
            project_storage=finished_experiment.project_storage,
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
