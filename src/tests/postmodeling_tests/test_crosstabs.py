

import pandas as pd
import pytest 

# from triage.component.postmodeling.crosstabs import run_crosstabs, run_crosstabs_from_matrix
from sqlalchemy import text
from triage.database_reflection import table_has_data
from triage.component.postmodeling.base import (
    SingleModelAnalyzer,
)
from triage.component.postmodeling.crosstabs import run_crosstabs_from_matrix

def test_run_crosstabs(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    project_path = finished_experiment.project_storage.project_path
    # mock test has 33 predictions
    thresholds = {'rank_abs_no_ties': 10}

    # 1. verify that there are no crosstabs for this model 
    assert not table_has_data('test_results.crosstabs', db_engine)
    
    # 2. generate crosstabs  
    
    sma = SingleModelAnalyzer(model_id, db_engine)
    crosstabs_df = sma.crosstabs_pos_vs_neg(project_path, thresholds)
    
    # 3. verify that we have data in crosstabs table 
    assert table_has_data('test_results.crosstabs', db_engine)

    # 4. verify that crosstabs table has the crosstabs for model_id 1 
    q = '''
        select count(*)
        from test_results.crosstabs
        where model_id = :model_id
    '''
    with db_engine.connect() as conn:
        result = conn.execute(
            text(q),
            {'model_id': model_id}
        ).first()
    count = result[0]
    assert count > 0

    
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
            and threshold <= {threshold}
        '''
        
        df = pd.read_sql(q, finished_experiment.db_engine)
                
        if df.empty:
            errors.append(threshold_type)
 
    assert not errors, f"errors occured for: {errors}"    


def test_run_crosstabs_no_predictions(finished_experiment_without_predictions):
    db_engine = finished_experiment_without_predictions.db_engine
    project_path = finished_experiment_without_predictions.project_storage.project_path
    model_id = 1
    thresholds = {'rank_abs_no_ties': 10}

    # verify that there are no predictions for this model 
    with finished_experiment_without_predictions.db_engine.connect() as conn:
        result = conn.execute(
            text("""
                select count(*)
                from test_results.predictions 
                where model_id = :model_id
            """),
            {'model_id': model_id}
        ).first()

    assert result[0] == 0

    sma = SingleModelAnalyzer(model_id, db_engine)

    with pytest.raises(ValueError):
        sma.crosstabs_pos_vs_neg(project_path, thresholds)                                                
                    

def test_run_crosstabs_invalid_matrix_uuid(finished_experiment):
    """assert that an invalid matrix uuid raises a ValueError"""
    db_engine = finished_experiment.db_engine
    project_path = finished_experiment.project_storage.project_path
    model_id = 1
    thresholds = {'rank_abs_no_ties': 10}   
    matrix_uuid = 'invalid_uuid'

    sma = SingleModelAnalyzer(model_id, db_engine)
    
    with pytest.raises(ValueError):
        sma.crosstabs_pos_vs_neg(project_path, thresholds, matrix_uuid=matrix_uuid)


def test_run_crosstabs_different_table_name(finished_experiment): 
    db_engine = finished_experiment.db_engine
    project_path = finished_experiment.project_storage.project_path
    model_id = 1
    thresholds = {'rank_abs_no_ties': 10}
    table_name = 'crosstabs_test_table'

    # 1. verify that there are no predictions for this model 
    assert not table_has_data(f'test_results.{table_name}', db_engine)

    # 2. generate crosstabs
    sma = SingleModelAnalyzer(model_id, db_engine)
    sma.crosstabs_pos_vs_neg(project_path, thresholds, table_name=table_name, return_df=False)  

    # 3. verify that we have data in crosstabs table
    assert table_has_data(f'test_results.{table_name}', db_engine)


def test_run_crosstabs_from_matrix_delete_table_if_exists(finished_experiment): 
    db_engine = finished_experiment.db_engine
    project_path = finished_experiment.project_storage.project_path
    model_id = 1
    thresholds = {'rank_abs_no_ties': 10}

    # 1. add crosstabs 
    sma = SingleModelAnalyzer(model_id, db_engine)
    sma.crosstabs_pos_vs_neg(project_path, thresholds, return_df=False)

    # 2. verify that we have data in crosstabs table
    assert table_has_data('test_results.crosstabs', db_engine)

    # 3. delete crosstabs table if exists and re-add crosstabs
    for threshold_type, threshold in thresholds.items():
        df = run_crosstabs_from_matrix(
            db_engine=db_engine,
            project_path=project_path,
            model_id=model_id,
            threshold_type=threshold_type,
            threshold=threshold,
            replace=True,
        )

    # 4. verify that we have data in crosstabs table
    assert table_has_data('test_results.crosstabs', db_engine)


