from triage.component.postmodeling.crosstabs import run_crosstabs
from triage.database_reflection import table_has_data


def test_run_crosstabs(finished_experiment, crosstabs_config):
    run_crosstabs(finished_experiment.db_engine, crosstabs_config)
    expected_table_name = (
        crosstabs_config.output["schema"] + "." + crosstabs_config.output["table"]
    )
    table_has_data(expected_table_name, finished_experiment.db_engine)
