import pandas as pd
import numpy as np
import yaml
import os
import psycopg2

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.tree import export_text

from triage.component.catwalk.storage import ProjectStorage


def _get_error_analysis_configuration():
    """
        Return:
            Dictionary of error analysis configuration
    """
    # TODO need to change it as in 
    filename_yaml = os.path.join(
        os.path.abspath(os.getcwd()), 
        "src/triage/component/postmodeling/config.yaml")
    with open(filename_yaml, 'r') as f:
        config = yaml.full_load(f)
    error_analysis_config = config['error_analysis']

    return error_analysis_config


def _get_random_seed(model_id, db_conn):
    """
        Args: 
            model_id (int): The model id from to retrieve the random seed for.
            db_conn (sqlalchemy.engine.connect): Simple connection to the database.

        Returns:
            Integer of the random seed associated to this model_id.
    """
    q = """
        SELECT 
            random_seed
        FROM triage_metadata.models
        WHERE model_id = {model_id}
    """.format(model_id=model_id)

    random_seed = pd.read_sql(q, db_conn)

    return random_seed.random_seed.values[0]


def fetch_scores_labels(model_id, db_conn):
    """
    Given a model id, it retrieves its scores and labels.
    
    Args: 
        model_id (int): The model id from which scores and labels are going to be retrieved from. 
        db_conn (sqlalchemy.engine.connect): Simple connection to the database.
    
    Returns: 
        DataFrame with scores and labels for a specific model_id.
    """

    q = """
        SELECT 
           model_id,
           entity_id,
           as_of_date,
           score,
           label_value,
           rank_abs_no_ties,
           matrix_uuid
        FROM test_results.predictions 
        WHERE model_id = {model_id}
    """.format(model_id=model_id)
    
    df = pd.read_sql(q, db_conn)
    df.set_index(['entity_id', 'as_of_date'], inplace=True)

    return df


def fetch_matrices(model_id, project_path, db_conn):
    """
    Args: 
        model_id ():
        project_path ():
        db_conn ():

    Returns:
        A DataFrame with features, predictions, and labels
    """
    # getting predictions with label and matrix_uuid 
    predictions = fetch_scores_labels(model_id, db_conn)
    matrix_uuid = predictions.matrix_uuid.unique()[0]

    project_storage = ProjectStorage(project_path)
    matrix_storage_engine = project_storage.matrix_storage_engine()

    matrix_store = matrix_storage_engine.get_store(matrix_uuid=matrix_uuid)
    matrix = matrix_store.design_matrix
    #features = list(matrix.columns)

    # joining the predictions and labels for error analysis
    matrix_w_preds = predictions.join(matrix, how='left')

    return matrix_w_preds


def generate_error_labels(matrix, k):
    """
    Adds columns that identify the different errors the model made by looking only into FPs, FNs or both; 
    
    Args:
        matrix (DataFrame): DataFrame with the scores and labels.
        k (int): Number of resources to use
    
    Returns: 
        DataFrame with two new columns associated to the labels generated for each type of error: general error, 
        errors in positive label, errors in negative labels.
    """

    # sort the scores desc
    sorted_scores = matrix.sort_values(by="rank_abs_no_ties")
    # add prediction column
    sorted_scores['prediction'] = '0'
    sorted_scores.loc[sorted_scores.rank_abs_no_ties <= k, 'prediction'] = '1'
    # add type of label: TP, TN, FP, FN
    sorted_scores['type_label'] = np.where(~(sorted_scores.label_value) & (sorted_scores.prediction == '1'), 'FP', 'TP')
    sorted_scores['type_label'] = np.where((sorted_scores.label_value) & (sorted_scores.prediction == '0'), 'FN', sorted_scores.type_label)
    sorted_scores['type_label'] = np.where(~(sorted_scores.label_value) & (sorted_scores.prediction == '0'), 'TN', sorted_scores.type_label)
    
    # add three new columns with error analysis labels
    sorted_scores['error_negative_label'] = np.where(sorted_scores.type_label == 'FN', '1', '0')
    sorted_scores['error_positive_label'] = np.where(sorted_scores.type_label == 'FP', '1', '0')
    sorted_scores['error_general'] = np.where((sorted_scores.type_label == 'FP') | 
                                       (sorted_scores.type_label == 'FN'), '1', '0')

    return sorted_scores


def error_analysis_model(model_id, matrix, grid, k, random_seed):
    """

        Args: 
            model_id (int):
            matrix (DataFrame):
            grid (ParamGrid): 

        Returns:
            List of dictionaries with results from all decision trees generated for 
            each configuration on each error type analysis
    """
    error_analysis_types = [element for element in list(matrix.columns) 
                                if element.startswith('error')]

    error_analysis_results = []
    for error_type in error_analysis_types:
        results = {'model_id': model_id, 'error_type': error_type, 'k': k}

        # first 5 columns of matrix aren't features
        predictions_cols = ['model_id', 'score', 'label_value', 
                            'rank_abs_no_ties', 'matrix_uuid', 'prediction']
        no_features = predictions_cols + ['type_label'] + error_analysis_types

        if error_type == 'error_negative_label':
            X = matrix[matrix.type_label.isin(['TN', 'FN'])]\
                .drop(no_features, axis=1)
            y = matrix[matrix.type_label.isin(['TN', 'FN'])]\
                .filter([error_type], axis=1)
        elif error_type == 'error_positive_label':
            X = matrix[matrix.type_label.isin(['TP', 'FP'])]\
                .drop(no_features, axis=1)
            y = matrix[matrix.type_label.isin(['TP', 'FP'])]\
                .filter([error_type], axis=1)
        else:
            X = matrix.drop(no_features, axis=1)
            y = matrix.filter([error_type], axis=1)
       
        for config in ParameterGrid(grid): 
            config_params = {}
            max_depth = config['max_depth']
            dt = DecisionTreeClassifier(max_depth=max_depth, 
                random_state=random_seed)
            error_model = dt.fit(X, y)
            feature_importances_ = error_model.feature_importances_
            # TODO top n of feature importances should be a parameter 
            importances_idx = list(np.argsort(-feature_importances_)[:10])
            feature_ = list(X.columns)
            feature_names_importance_sorted = [feature_[element] for element in importances_idx]

            config_params['max_depth'] = max_depth
            config_params['feature_importance'] = feature_importances_[list(np.argsort(-feature_importances_)[:10])]
            config_params['feature_names'] = feature_names_importance_sorted
            config_params['tree_text'] = export_text(error_model, feature_names=feature_)

            error_analysis_results.append(results | config_params)

    return error_analysis_results


def error_analysis(model_id, db_conn):
    """
    
        Args: 
            model_group_ids (list): List of model groups ids 
            db_conn (): Database engine connection
    """
    error_analysis_config = _get_error_analysis_configuration()
    project_path = error_analysis_config['project_path']
    k_set = error_analysis_config['k']
    grid = error_analysis_config['model_params']
    
    matrix_data = fetch_matrices(model_id, project_path, db_conn)
    error_analysis_results = []
    for k in k_set:
        new_matrix = generate_error_labels(matrix_data, k)
        random_seed = _get_random_seed(model_id, db_conn)
        error_analysis_result = error_analysis_model(model_id, new_matrix, grid, k, random_seed)
        error_analysis_results.append(error_analysis_result)

if __name__ == "__main__":
    model_id = 236
    db_conn = psycopg2.connect(service='acdhs_housing')
    
    error_analysis(model_id, db_conn)