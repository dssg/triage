from triage.component.risklist import generate_risk_list
from tests.utils import sample_config, populate_source_data
from triage.experiments import SingleThreadedExperiment
from triage.validation_primitives import table_should_have_data


def test_risklist(db_engine, project_storage):
    # given a model id and as-of-date <= today
    # and the model id is trained and is linked to an experiment with feature and cohort config
    # generate records in listpredictions
    # the # of records should equal the size of the cohort for that date
    populate_source_data(db_engine)
    SingleThreadedExperiment(
        sample_config(),
        db_engine=db_engine,
        project_path=project_storage.project_path
    ).run()

    model_id = 1
    as_of_date = '2013-01-01'
    generate_risk_list(db_engine, model_id, as_of_date)
    table_should_have_data(
        db_engine=db_engine,
        table_name="production.list_predictions",
    )
