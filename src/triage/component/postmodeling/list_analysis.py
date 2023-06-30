""" This is a temporaty script for writing functions that analyse the top-k lists,
    while others are working on other components. Eventually these files will be merged
"""

import logging
import itertools
import pandas as pd

# remove this
import psycopg2

from scipy.stats import spearmanr

from triage.component.catwalk.storage import ProjectStorage


# TODO: Modify how the threshold is specified
def get_highest_risk_entities(db_engine, model_id, threshold, include_all_ties=False):
    """ Fetch the entities with the highest risk for a particular model
        
        args:
            db_engine (sqlalchemy.engine)   : Database connection engine
            model_id (int)  : Model ID we are interested in
            threshold (Union[int, float]): value of the threshold (k)
            include_all_ties (bool): Whether to include all entities with the same score in the list (can include more than k elements) 
                                    Defaults to False where ties are broken randomly

        return:
            a pd.DataFrame object that contains the entity_id, as_of_date, the hash of the text matrix, model score, 
            and all rank columns 
            
    """
    
    # Determining which rank column to filter by
    ties_suffix = 'no_ties'
    col = 'rank_pct'
    
    if include_all_ties:
        ties_suffix='with_ties'

    if isinstance(threshold, int):
        col = 'rank_abs'      

    col_name = f'{col}_{ties_suffix}'

    q = f"""
        select 
            entity_id, 
            as_of_date,
            score,
            label_value,
            rank_abs_no_ties,
            rank_abs_with_ties,
            rank_pct_no_ties,
            rank_pct_with_ties,
            matrix_uuid
        from test_results.predictions
        where model_id={model_id}
        and {col_name} <= {threshold}
    """

    top_k = pd.read_sql(q, db_engine)

    return top_k
   

def pairwise_comparison_model_lists(db_engine, models, k, include_all_tied_entities=False):
    """ Given two lists compare their similarities

        args:
            db_engine (sqlalchemy.engine): Database connection engine 
            models (List[int]): A list of model_ids that we are interested in comparing
            k (int): The list size
            include_all_tied_entities (bool): Whether to include all entities with the same score in the list 
                        If True, can include more than k elements in the list.  
                        Defaults to False where ties are broken randomly
 
        return:
            pairwise list comparison metrics (Dict[Dict]): Dictionay of dictionaries 
                                that contains 'overlap', 'jaccard_similarity', 'rank_corr' for each model_id pair
    """
    
    logging.info('Fetching the top-k lists for all models')
    risk_lists = dict()
    for model_id in models:
        risk_lists[model_id] = get_highest_risk_entities(db_engine, model_id, k, include_all_tied_entities)

    pairs = list(itertools.combinations(risk_lists.keys(), 2))

    logging.info(f'You provided {len(models)} models. Performing {len(pairs)} comparisons')

    # Storing all results as a dictionary of dictionaries
    # Key model_id pair-- Tuple[(int, int)], Value -- Dict[jaccard, overlap, rank_corr]
    results = dict()

    for model_pair in pairs:
        logging.info(f'Comparing {model_pair[0]} and {model_pair[1]}')

        # Dictionary that stores the comparison metrics 
        d = dict()
    
        df1 = risk_lists[model_pair[0]]
        df2 = risk_lists[model_pair[1]]

        if df1.as_of_date.at[0] != df2.as_of_date.at[0]:
            logging.warning('You are comparing two lists generated on different as_of_dates!')

        # calculating jaccard similarity and overlap
        entities_1 = set(df1.entity_id)
        entities_2 = set(df2.entity_id)

        inter = entities_1.intersection(entities_2)
        un = entities_1.union(entities_2)    
        d['jaccard'] = len(inter)/len(un)

        # If the list sizes are not equal, using the smallest list size to calculate simple overlap
        d['overlap'] = len(inter)/ min(len(entities_1), len(entities_2))

        # calculating rank correlation
        df1.sort_values('score', ascending=False, inplace=True)
        df2.sort_values('score', ascending=False, inplace=True)

        # only returning the corr coefficient, not the p-value
        d['rank_corr'] = spearmanr(df1.entity_id.iloc[:, 0], df2.entity_id.iloc[:, 1])[0]

        return results


def _fetch_relevant_matrix_hashes(db_engine, model_id):
    """"""

    q = f"""
        select 
            train.matrix_uuid as train_matrix_uuid,
            test.matrix_uuid as test_matrix_uuid
        from train_results.prediction_metadata train join test_results.prediction_metadata test
        using(model_id) 
        where model_id={model_id};
    """

    matrices = pd.read_sql(q, db_engine).to_dict(orient='records')

    return matrices


# Currently this function only calculates the mean ratio
# NOTE This code was incorported to the ModelAnalyzer
def get_crosstabs_postive_vs_negative(
    db_engine, 
    model_id, 
    project_path, 
    thresholds,  
    return_df=True, 
    matrix_uuid=None, 
    table_name='crosstabs'
):
    """ For a given model_id, generate crosstabs for the top_k vs the rest
    
        args:
            model_id (int): The model_id we are intetereted in
            project_path (str): Path where the experiment artifacts (models and matrices) are stored
            thresholds (Dict{str: Union[float, int}]): A dictionary that maps threhold type to the threshold
                                                    The threshold type can be one of the rank columns in the test_results.predictions_table
            return_df (bool, optional): Whether to return the constructed df or just to store in the database
                                        Defaults to False (only writing to the db)
            table_name (str, optional): Table name to use in the db's `test_results` schema. Defaults to crosstabs
            matrix_uuid (str, optional): If we want to run crosstabs for a different matrix than the validation matrix from the experiment
    """

    # Table structure
    # model_id, matrix_uuid, 
    # threshold_type, threshold, feature, metric, value
    

    if matrix_uuid is None:
        matrix_uuid = _fetch_relevant_matrix_hashes(db_engine, model_id)[0]['test_matrix_uuid']

    logging.info('Fetching predictions for the model')
    
    # NOTE/TODO If we use a Model object here, we can avoid these repeated db calls
    q = f"""
        select 
            entity_id, 
            as_of_date,
            score,
            label_value,
            rank_abs_no_ties,
            rank_abs_with_ties,
            rank_pct_no_ties,
            rank_pct_with_ties,
            matrix_uuid
        from test_results.predictions
        where model_id={model_id} and matrix_uuid = '{matrix_uuid}'
    """

    predictions = pd.read_sql(q, db_engine)
    predictions.set_index(['entity_id', 'as_of_date'], inplace=True)

    if predictions.empty:
        logging.error(f'No predictions found for {model_id} and matrix {matrix_uuid}. Exiting!')
        raise ValueError(f'No predictions found {model_id} and matrix {matrix_uuid}')


    # initializing the storage engines
    project_storage = ProjectStorage(project_path)
    matrix_storage_engine = project_storage.matrix_storage_engine()

    matrix_store = matrix_storage_engine.get_store(matrix_uuid=matrix_uuid)
    matrix = matrix_store.design_matrix
    labels = matrix_store.labels
    features = matrix.columns

    # joining the predictions to the model
    matrix = predictions.join(matrix, how='left')

    for threshold_name, threshold in thresholds.items():
        logging.debug('')
        msk = matrix[threshold_name] <= threshold
        postive_preds = matrix[msk]
        negative_preds = matrix[~msk]

        # TODO: Take a list of metrics to calculate and iterate (as it's done in crosstabs.py)

        # Calculates the mean ratio for each feature and produces a series indexed by the feature n,e
        mean_ratios = (postive_preds[features].mean() / negative_preds[features].mean()).reset_index()
        mean_ratios['metric'] = 'mean_ratio_pos_over_neg'
        
        non_zero_rows_count_pos_pred = (postive_preds[features] > 0).sum().reset_index()
        non_zero_rows_count_pos_pred['metric'] = 'non_zero_rows_pos_pred_count'

        non_zero_rows_frac_pos_pred = (postive_preds[features] > 0).mean().reset_index()
        non_zero_rows_frac_pos_pred['metric'] = 'non_zero_rows_pos_pred_pct'
        
        non_zero_rows_count_neg_pred = (negative_preds[features] > 0).sum().reset_index()
        non_zero_rows_count_neg_pred['metric'] = 'non_zero_rows_neg_pred_count'

        non_zero_rows_frac_neg_pred = (negative_preds[features] > 0).mean().reset_index()
        non_zero_rows_frac_neg_pred['metric'] = 'non_zero_rows_pos_pred_pct'

        crosstabs_df = pd.concat([
            mean_ratios, 
            non_zero_rows_count_pos_pred, 
            non_zero_rows_count_neg_pred, 
            non_zero_rows_frac_pos_pred,
            non_zero_rows_frac_neg_pred
        ])

        crosstabs_df.rename(columns={'index': 'feature', 0: 'value'}, inplace=True)
        crosstabs_df['model_id'] = model_id
        crosstabs_df['matrix_uuid'] = matrix_uuid
        crosstabs_df['threshold_type'] = threshold_name
        crosstabs_df['threshold'] = threshold

        crosstabs_df.to_csv('temp_crosstabs.csv', index=False)

        if return_df:
            return crosstabs_df
        


def _get_descriptives(db_engine, model_id, columns_of_interest, feature_groups):
    """ Given a list of entities, generated descriptives

        args:
            entities (pd.DataFrame): The Dataframe containing information of the entities
            columns_of_interest (List[str]): The list of column names we are interested in describing
    """

    # NOTE: As the first pass, we'll calculate descriptives of 


if __name__ == '__main__':

    conn = psycopg2.connect(service='acdhs_housing')

    df = get_crosstabs_postive_vs_negative(
        conn,
        model_id=1533,
        project_path='s3://dsapp-social-services-migrated/acdhs_housing/triage_experiments/',
        thresholds={'rank_abs_no_ties': 100},
        table_name='crosstabs_test_kasun'
    )

    print(df.sample(n=10))