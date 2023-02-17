import pandas as pd
import numpy as np
import yaml
import os
import psycopg2
import verboselogs, logging, coloredlogs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.tree import export_text

from triage.component.catwalk.storage import ProjectStorage

logger = verboselogs.VerboseLogger(__name__)

CONFIG_PATH = os.path.join(os.getcwd(),
'/mnt/data/users/lily/triage/src/triage/component/postmodeling/config.yaml')

def _get_error_analysis_configuration(path):
    """
        Return:
            Dictionary of error analysis configuration
    """
    # TODO need to change it as in the rest of triage
    filename_yaml = path
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
    logger.debug("""obtained random seed associated to model_id: {model_id} 
    for generating decision trees in error analysis""")

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
    logger.debug(f"predictions and labels joined for error analysis of model_id {model_id}")

    return matrix_w_preds


def generate_error_labels(matrix, k):
    """
    Add columns to the original DataFrmae to define labels for the error model analysis.

    Args:
        matrix (DataFrame): DataFrame with the scores and labels.
        k (int): Number of resources to use
    
    Returns: 
        DataFrame with two new columns associated to the labels generated for each type of error: general error, 
        errors in positive label, errors in negative labels.
    """
    # extra df to create additional columns to the existing DF efficiently  
    # since the DF is "highly fragmentated" according to pandas.  
    prediction_col = ['0' for element in range(matrix.shape[0])]
    type_label_col = ['TP' for element in range(matrix.shape[0])]
    error_negative_label = ['0' for element in range(matrix.shape[0])]
    error_positive_label = ['0' for element in range(matrix.shape[0])]
    error_general_label = ['0' for element in range(matrix.shape[0])]

    # sort the scores desc
    sorted_scores = matrix.sort_values(by="rank_abs_no_ties")
    
    indexes = sorted_scores.index
    extra_df = pd.DataFrame({'prediction': prediction_col,
                            'type_label': type_label_col, 
                            'error_negative_label': error_negative_label,
                            'error_positive_label': error_positive_label,
                            'error_general_label': error_general_label 
                            }, index=indexes)
    
    data_df = pd.concat([sorted_scores, extra_df], axis=1)
    # add prediction column
    #sorted_scores['prediction'] = '0'
    data_df.loc[data_df.rank_abs_no_ties <= k, 'prediction'] = '1'
    # add type of label: TP, TN, FP, FN
    data_df['type_label'] = np.where(~(data_df.label_value) & 
                                    (data_df.prediction == '1'), 'FP', 'TP')
    data_df['type_label'] = np.where((data_df.label_value) & 
                                    (data_df.prediction == '0'), 'FN', 
                                    data_df.type_label)
    data_df['type_label'] = np.where(~(data_df.label_value) & (data_df.prediction == '0'), 'TN', data_df.type_label)
    
    # add three new columns with error analysis labels
    data_df['error_negative_label'] = np.where(data_df.type_label == 'FN', '1', '0')
    data_df['error_positive_label'] = np.where(data_df.type_label == 'FP', '1', '0')
    data_df['error_general_label'] = np.where((data_df.type_label == 'FP') | 
                                       (data_df.type_label == 'FN'), '1', '0')
    logger.debug(f"error labels generated for k {k}")

    return data_df


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
            logger.debug(f"error analysis generated for model_id {model_id}, with k {k}")

    return error_analysis_results


def generate_error_analysis(model_id, db_conn):
    """
    
        Args: 
            model_group_ids (list): List of model groups ids 
            db_conn (): Database engine connection
        
        Returns: 
            List of list of dictionaries 
    """
    error_analysis_config = _get_error_analysis_configuration(CONFIG_PATH)
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
    #TODO where and how to save the outputs of the errror analysis (list of dicts)
    logger.debug(f"error analysis finished for model id {model_id}")

    return error_analysis_results


def plot_feature_importance(importances, features):
    """
    Plots a seborn barplot with the most important features associated to the 
    error in the label analyzed.

        Args: 
            importances (list): List with the importance value of the top n 
                                important features
            features (list): List with the name of the n most important features 
    """
    plt.clf()
    df_viz = pd.DataFrame({'importance': importances, 'feature': features})
    sns.barplot(data=df_viz.sort_values(by="importance", ascending=False), x="importance", y="feature",
               color="grey")
    plt.show()


def _output_config_text(element, error_label):    
    config_text = f"""

    Error analysis type: {error_label}, size of the list: {element['k']}

    Decision Tree with max_depth of, {element['max_depth']}

    Top 10 features associated with error in label type {error_label}
    """
    
    return config_text


def _output_dt_text(error_label):
    dt_text = f"""
    Rules made with the top 10 features associated with errors in label type {error_label}
    """
    return dt_text
  

def _get_error_label(element):
    if element['error_type'] == 'error_negative_label': 
        error_label = 'Negative (FN)'
    elif element['error_type'] == 'error_positive_label':
        error_label = 'Positive (FP)'
    else: 
        error_label = 'Positive and Negative (FP & FN)'

    return error_label


def output_all_analysis(error_analysis_results):
    print("Error Analysis")
    print("--------------")

    for analysis in error_analysis_results:
        for element in analysis:
            error_label = _get_error_label(element)
            config_text = _output_config_text(element, error_label)
            print(config_text)
            
            plot_feature_importance(element['feature_importance'], 
                                    element['feature_names'])

            dt_text = _output_dt_text(error_label)
           
            print(element['tree_text'])
            print("             ######            ")
        
        print("*******************************************")


def output_specific_configuration(error_analysis_resutls, k=100, max_depth=10,
                                    error_type="error_negative_label"):
    """
    """
    for analysis in error_analysis_resutls:
        for element in analysis:
            if ((element['error_type'] == error_type) &
            (element['k'] == k) & 
            (element['max_depth'] == max_depth)):
                error_label = _get_error_label(element)
                config_text = _output_config_text(element, error_label)
                print(config_text)
                
                plot_feature_importance(element['feature_importance'], 
                                        element['feature_names'])

                dt_text = _output_dt_text(error_label)
            
                print(element['tree_text'])
                print("             ######            ")



def output_specific_error_analysis(error_analysis_results, 
                                      error_type='error_negative_label'):
    """
    """
    for analysis in error_analysis_results:
        for element in analysis:
            if element['error_type'] == error_type:
                error_label = _get_error_label(element)
                config_text = _output_config_text(element, error_label)
                print(config_text)
                
                plot_feature_importance(element['feature_importance'], 
                                        element['feature_names'])

                dt_text = _output_dt_text(error_label)
            
                print(element['tree_text'])
                print("             ######            ")
        
        print("*******************************************")



if __name__ == "__main__":
    model_id = 1630
    db_conn = psycopg2.connect(service='acdhs_housing')

    generate_error_analysis(model_id, db_conn)