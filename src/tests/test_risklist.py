from triage.risklist import generate_risk_list, train_matrix_info_from_model_id, experiment_config_from_model_id
from triage.validation_primitives import table_should_have_data


def test_risklist(finished_experiment):
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


def test_experiment_config_from_model_id(finished_experiment):
    model_id = 1
    experiment_config = experiment_config_from_model_id(finished_experiment.db_engine, model_id)
    assert experiment_config == finished_experiment.config


def test_train_matrix_info_from_model_id(finished_experiment):
    model_id = 1
    (train_matrix_uuid, matrix_metadata) = train_matrix_info_from_model_id(finished_experiment.db_engine, model_id)
    assert train_matrix_uuid
    assert matrix_metadata
