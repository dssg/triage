import pandas as pd

def retrieve_scores_labels(model_id, db_conn):
    """
    Retrieves the scores and labels associated to a particular model id.
    
    Args: 
        model_id (int): The model id from which scores and labels are going to be retrieved from. 
    
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
        select 
           model_id,
           entity_id, 
           as_of_date,
           score,
           label_value
        from test_results.predictions
        where model_id = {model_id}
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


def retrieve_features_matrix(model_id, path, storage="s3"):
    pass