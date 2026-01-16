

import pandas as pd

from triage.component.postmodeling.crosstabs import run_crosstabs, run_crosstabs_from_matrix
from triage.database_reflection import table_has_data


def test_run_crosstabs(finished_experiment, crosstabs_config):
    run_crosstabs(finished_experiment.db_engine, crosstabs_config)
    expected_table_name = (
        crosstabs_config.output["schema"] + "." + crosstabs_config.output["table"]
    )
    assert table_has_data(expected_table_name, finished_experiment.db_engine)

    
def test_run_crosstabs_from_matrix(finished_experiment):
    """assert that crosstabs table is populated for all threshold types"""
    
    thresholds = {
        'rank_abs_no_ties': 50,
        'rank_abs_with_ties': 50, 
        'rank_pct_no_ties': 0.1,
        'rank_pct_with_ties': 0.1
    }
    table_schema='test_results'
    table_name = 'crosstabs'
    
    errors = list()
    for threshold_type, threshold in thresholds.items():    
        df = run_crosstabs_from_matrix(
            db_engine=finished_experiment.db_engine,
            project_path=finished_experiment.project_path,
            model_id=1,
            threshold_type=threshold_type,
            threshold=threshold,
            push_to_db=True,
            table_schema=table_schema,
            table_name=table_name,
            replace=True
        )
        
        q = f'''
            select 
            1
            from {table_schema}.{table_name}
            where model_id = 1
            and threshold_type = '{threshold_type}'
            and threshold = {threshold}
        '''
        
        df = pd.read_sql(q,finished_experiment.db_engine)
                
        if df.empty:
            errors.append(threshold_type)
 
    assert not errors, f"errors occured for: {errors}"    
