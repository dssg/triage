import pytest 

from triage.component.postmodeling.crosstabs import run_crosstabs, run_crosstabs_from_matrix
from triage.database_reflection import table_has_data


def test_run_crosstabs(finished_experiment, crosstabs_config):
    run_crosstabs(finished_experiment.db_engine, crosstabs_config)
    expected_table_name = (
        crosstabs_config.output["schema"] + "." + crosstabs_config.output["table"]
    )
    table_has_data(expected_table_name, finished_experiment.db_engine)


def test_run_crosstabs_from_matrix(finished_experiment):
    """assert that crosstabs table is populated for all threshold types"""
    project_path = finished_experiment.project_storage.project_path
    engine=finished_experiment.db_engine
    model_id = 1
    thresholds = {
        'rank_abs_no_ties': 50,
        'rank_abs_with_ties': 50, 
        'rank_pct_no_ties': 0.1,
        'rank_pct_with_ties': 0.1
    }
    table_schema='test_results'
    table_name = 'crosstabs'
    
    for threshold_type, threshold in thresholds.items():
        run_crosstabs_from_matrix(
            db_engine=engine,
            project_path=project_path,
            model_id=model_id,
            threshold_type=threshold_type,
            threshold=threshold,
            push_to_db=True,
            table_schema=table_schema,
            table_name=table_name
        )
        
        assert table_has_data(f'{table_schema}.{table_name}', engine)
    