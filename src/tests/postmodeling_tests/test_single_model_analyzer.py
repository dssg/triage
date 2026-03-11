import pandas as pd
import pytest 

# from triage.component.postmodeling.crosstabs import run_crosstabs, run_crosstabs_from_matrix
from sqlalchemy import text
from triage.database_reflection import table_has_data
from triage.component.postmodeling.base import (
    SingleModelAnalyzer,
)
from triage.component.postmodeling.crosstabs import run_crosstabs_from_matrix


def test_get_predictions(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    
    # mock test has 33 predictions
    sma = SingleModelAnalyzer(model_id, db_engine)
    predictions_df = sma.get_predictions()
    assert isinstance(predictions_df, pd.DataFrame)
    assert predictions_df.shape[0] == 33


def test_get_top_k(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    thresholds = {
        'rank_abs_no_ties': 5,
        'rank_abs_with_ties': 5, 
        'rank_pct_no_ties': 0.1, 
        'rank_pct_with_ties': 0.5,
    }

    sma = SingleModelAnalyzer(model_id, db_engine)
    for threshold_type, threshold in thresholds.items():            
        top_k_df = sma.get_top_k(threshold_type, threshold)
        assert isinstance(top_k_df, pd.DataFrame)
        assert top_k_df.shape[0] > 0


def test_get_aequitas(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1

    sma = SingleModelAnalyzer(model_id, db_engine)
    aequitas_df = sma.get_aequitas()
    assert isinstance(aequitas_df, pd.DataFrame)
    assert aequitas_df.shape[0] > 0


def test_get_evaluations(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    
    sma = SingleModelAnalyzer(model_id, db_engine)
    evaluations_df = sma.get_evaluations()
    assert isinstance(evaluations_df, pd.DataFrame)
    assert evaluations_df.shape[0] > 0


def test_get_feature_importances(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    
    sma = SingleModelAnalyzer(model_id, db_engine)
    fi_df = sma.get_feature_importances()
    assert isinstance(fi_df, pd.DataFrame)
    assert fi_df.shape[0] > 0


def test_get_feature_group_importances(finished_experiment):
    db_engine = finished_experiment.db_engine
    model_id = 1
    
    sma = SingleModelAnalyzer(model_id, db_engine)
    fgi_df = sma.get_feature_group_importances()
    assert isinstance(fgi_df, pd.DataFrame)
    assert fgi_df.shape[0] > 0


def test_error_analysis(finished_experiment, postmodeling_config):
    pass
    # db_engine = finished_experiment.db_engine
    # project_path = finished_experiment.project_storage.project_path
    # model_id = 1
    
    # sma = SingleModelAnalyzer(model_id, db_engine)
    # ea = sma.error_analysis(project_path)
    # assert len(ea) > 0  