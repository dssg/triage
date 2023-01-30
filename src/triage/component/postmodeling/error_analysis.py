import pandas as pd

def retrieve_scores_labels(model_id, db_conn):
    """
    Retrieves the scores and labels associated to a particular model id.
    
    Args: 
        model_id (int): The model id from which scores and labels are going to be retrieved from. 
        db_conn (sqlalchemy.engine.connect): Simple connection to the database.
    
    Returns: 
        DataFrame with scores and labels for a specific model_id.
    """

    q = """
        --select
        --    model_id,
        --    entity_id, 
        --    as_of_date,
        --    score,
        --    label_value
        --from train_results.predictions
        --where model_id = {model_id}
        --union all
        SELECT 
           model_id,
           entity_id, 
           as_of_date,
           score,
           label_value
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
        DataFrame with three new columns associated to the labels for general error analysis, 
        error analysis for recall and error analysis for precison.
    """

    # sort the scores desc 
    sorted_scores = predictions.sort_values(by="score", ascending=False)
    # add prediction column
    sorted_scores['prediction'] = '0'
    sorted_scores.prediction.iloc[:k] = '1'
    # add type of label: TP, TN, FP, FN
    sorted_scores['type_label'] = 'TP'
    sorted_scores.type_label.mask(~(sorted_scores.label) & (sorted_scores.prediction == '1'), 'FP', inplace=True)
    sorted_scores.type_label.mask((sorted_scores.label) & (sorted_scores.prediction == '0'), 'FN', inplace=True)
    sorted_scores.type_label.mask(~(sorted_scores.label) & (sorted_scores.prediction == '0'), 'TN', inplace=True)
    
    # add three new columns with error analysis labels
    sorted_scores['error_label_recall'] = '0'
    sorted_scores.error_label_recall.mask(sorted_scores.type_label == 'FN', '1', inplace=True)
    sorted_scores['error_label_precision'] = '0'
    sorted_scores.error_label_precision.mask(sorted_scores.type_label == 'FP', '1', inplace=True)
    sorted_scores['error_label_general'] = '0'
    sorted_scores.error_label_general.mask((sorted_scores.type_label == 'FP') | 
                                       (sorted_scores.type_label == 'FN'), '1', inplace=True)
        
    return sorted_scores


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