import pandas as pd

from src.triage.component.postmodeling.utils import _fetch_relevant_model_matrix_info
from src.triage.component.catwalk.utils import sort_predictions_and_labels
from src.triage.component.catwalk.storage import ProjectStorage


def fetch_scores_labels(model_id, db_conn):
    """
    Given a model id, it retrieves its scores and labels.
    
    Args: 
        model_id (int): The model id from which scores and labels are going to be retrieved from. 
        db_conn (sqlalchemy.engine.connect): Simple connection to the database.
    
    Returns: 
        DataFrame with scores and labels for a specific model_id.
    """
    # instead use _fetch_relevant_model_matrix_info from str/triage/components/postmodeling/utils/add_predictions.py
    matrix_info = _fetch_relevant_model_matrix_info(db_conn, model_groups, experiment_hashes)
    q = """
        select
           model_id,
           entity_id, 
           'train' as type,
           as_of_date,
           score,
           label_value,
           rank_abs_no_ties,
           matrix_uuid
        from train_results.predictions
        where model_id = {model_id}
        union all
        SELECT 
           model_id,
           entity_id,
           'test' as type,
           as_of_date,
           score,
           label_value,
           rank_abs_no_ties,
           matrix_uuid
        FROM test_results.predictions
        WHERE model_id = {model_id}
    """.format(model_id=model_id)
    
    df = pd.read_sql(q, db_conn)
    
    return df


def generate_error_labels(predictions, k):
    """
    Adds columns that identify the different errors the model made by looking only into FPs, FNs or both; 
    
    Args:
        predictions (DataFrame): DataFrame with the scores and labels.
        k (int): Number of resources to use
    
    Returns: 
        DataFrame with two new columns associated to the labels generated for each type of error: general error, 
        errors in positive label, errors in negative labels.
    """

    # sort the scores desc
    sorted_scores = predictions.sort_values(by="rank_abs_no_ties")
    # add prediction column
    sorted_scores['prediction'] = '0'
    sorted_scores[sorted_scores.rank_abs_no_ties <= k] = 1
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
    df = sorted_scores.melt(id_vars=['model_id', 'entity_id', 'type', 'as_of_date', 'score', 'label_value',
    'rank_abs_no_ties', 'matrix_uuid', 'prediction', 'type_label'], value_vars=['error_negative_label', 
    'error_positive_label', 'error_general'], var_name="error_type", value_name="error_label")

    return df


def fetch_matrices(model_id, project_path):
   #_fetch_relevant_model_matrix_info()
    project_storage = ProjectStorage(project_path)
    matrix_storage_engine = project_storage.matrix_storage_engine()

    matrix_store = matrix_storage_engine.get_store(matrix_uuid=matrix_uuid)
    matrix = matrix_store.design_matrix
    labels = matrix_store.labels
    features = matrix.columns

    # joining the predictions to the model
    matrix = predictions.join(matrix, how='left')

    pass


def retrieve_features_matrix(model_id, db_conn, path, storage="s3"):
    """
    Retrieves the features matrix form an specific model id (train or test). 
    
    Args: 
        model_id (int): The model id from which the train features matrix is going to be retrieved.
        db_conn (sqlalchemy.engine.connect): Simple connection to database.
        storage (str): If the matrix is persisted in a S3 bucket or on disk. Possible values: s3, disk.
        path (str): Specific path where the matrix is located. In case of S3 it requires to start 
                    from the name of the bucket without the prefix s3://. 
    """
    
    q = """
        SELECT 
          model_id,
          model_hash,
          model_group_id,
          train_matrix_uuid as train_mat,
          matrix_uuid as test_mat
        FROM test_results.prediction_metadata
        JOIN triage_metadata.models using(model_id)
        WHERE model_id = {model_id}
    """.format(model_id=model_id)
    
    matrices_info = pd.read_sql(q, db_conn)
    test_feature_matrix = matrices_info.test_mat.values