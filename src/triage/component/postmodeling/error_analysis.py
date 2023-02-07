import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid

from triage.component.catwalk.storage import ProjectStorage
from triage.util.conf import get_error_analysis_configuration


def _get_model_ids(model_group, db_conn):
    q = """
        SELECT 
            distinct on (model_id) model_id, 
            model_group_id
        FROM triage_metadata.experiment_models a
        JOIN triage_metadata.models b using(model_hash)
            JOIN test_results.prediction_metadata c using(model_id)
        WHERE model_group_id in ({model_group})
    """.format(model_group=model_group)

    model_ids = pd.read_sql(q, db_conn)

    return list(model_ids.model_id)


def _get_random_seed(model_id, db_conn):
    q = """
        SELECT 
            random_seed
        FROM triage_metadata.models
        WHERE model_id = {model_id}
    """.format(model_id=model_id)

    random_seed = pd.read_sql(q, db_conn)

    return random_seed


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
    print(df.head())
    print("...scores and labels done")
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
    print("...glimpse predictions")
    print(predictions.head())
    print("...glimpse features matrix")
    print(matrix.head())
    matrix_w_preds = predictions.join(matrix, how='left')

    print("...matrix joined with predictions done")
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
    print(matrix.shape)
    sorted_scores = matrix.sort_values(by="rank_abs_no_ties")
    print("...glimpse os sorted df")
    print(sorted_scores.head())
    print("...sorted scores done")
    # add prediction column
    sorted_scores['prediction'] = '0'
    sorted_scores.loc[sorted_scores.rank_abs_no_ties <= k, 'prediction'] = '1'
    print("...prediction done")
    # add type of label: TP, TN, FP, FN
    sorted_scores['type_label'] = 'TP'
    sorted_scores.type_label.mask(~(sorted_scores.label_value) & (sorted_scores.prediction == '1'), 'FP', inplace=True)
    sorted_scores.type_label.mask((sorted_scores.label_value) & (sorted_scores.prediction == '0'), 'FN', inplace=True)
    sorted_scores.type_label.mask(~(sorted_scores.label_value) & (sorted_scores.prediction == '0'), 'TN', inplace=True)
    
    # add three new columns with error analysis labels
    sorted_scores['error_negative_label'] = '0'
    sorted_scores.error_negative_label.mask(sorted_scores.type_label == 'FN', '1', inplace=True)
    sorted_scores['error_positive_label'] = '0'
    sorted_scores.error_positive_label.mask(sorted_scores.type_label == 'FP', '1', inplace=True)
    sorted_scores['error_general'] = '0'
    sorted_scores.error_general.mask((sorted_scores.type_label == 'FP') | 
                                       (sorted_scores.type_label == 'FN'), '1', inplace=True)
    # melt columns to create error_type feature to use in DT
    #feature_names = list(matrix.columns.values)
    # first five names are from predictions matrix, not features
    #value_vars = ['error_negative_label', 
    #'error_positive_label', 'error_general']
    #id_variables = [element for element in feature_names if element not in value_vars]

    #df = sorted_scores.melt(id_vars=id_variables, value_vars=value_vars, 
    #var_name="error_type", value_name="error_label")

    print("...glimpse error labels")
    print(sorted_scores.head())
    print("...error labels generated done")
    return sorted_scores


def error_analysis_model(model_id, matrix, grid, k):
    """

        Args: 
            model_id (int):
            matrix (DataFrame):
            grid (ParamGrid): 

        Returns:
            Dictionary with results from all decision trees generated for 
            each configuration and each error type analysis
    """
    error_analysis_types = [element for element in list(matrix.columns) 
                                if element.startswith('error')]

    error_analysis_results = []
    for error_type in error_analysis_types:
        # TODO we are missing k in the results!
        results = {'model_id': model_id, 'error_type': error_type, 'k': k}

        # first 5 columns of matrix aren't features
        no_features = list(matrix.columns)[:5] + ['prediction', 'type_label'] + \
            [error_type]
       
        X = matrix.drop(no_features, axis=1)
        y = matrix.filter([error_type], axis=1)
       
        parameter_grid = list(ParameterGrid(grid))
        for config in parameter_grid: 
            dt = DecisionTreeClassifier(max_depth=config['max_depth'])
            error_model = dt.fit(X, y)
            feature_importances_ = error_model.feature_importances_
            # TODO top n of feature importances should be a parameter 
            importances_idx = list(np.argsort(feature_importances_)[-10:])
            feature_ = list(matrix.columns)
            feature_names_importance_sorted = [feature_[element] for element in importances_idx]

            results['max_depth'] = config['max_depth']
            results['feature_importance'] = error_model.feature_importances_
            results['feature_names'] = feature_names_importance_sorted
            results['rules'] = error_model.tree_

            error_analysis_results.append(results)
    
    print("...glimpse error analysis results")
    print(error_analysis_results)
    print("...error analysis done")
    return error_analysis_results


def error_analysis(model_group_ids, db_conn):
    """
    
        Args: 
            model_group_ids (list): List of model groups ids 
            db_conn (): Database engine connection
    """
    error_analysis_config = get_error_analysis_configuration()
    project_path = error_analysis_config['project_path']
    k_set = error_analysis_config['k']
    grid = error_analysis_config['model_params']

    for group_id in model_group_ids:
        model_ids = _get_model_ids(group_id, db_conn)
        for model_id in model_ids:
            matrix_data = fetch_matrices(model_id, project_path, db_conn)
            for k in k_set:
                print("k", k)
                new_matrix = generate_error_labels(matrix_data, k)
                error_analysis_model(model_id, new_matrix, grid, k)