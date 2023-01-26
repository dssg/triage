""" This is a temporaty script for writing functions that analyse the top-k lists,
    while others are working on other components. Eventually these files will be merged
"""

import logging
import pandas as pd

from scipy.stats import spearmanr


def get_highest_risk_k_entities(db_engine, model_id, k, fetch_all_ties=False):
    """ Fetch the entities with the highest risk for a particular model
        
        args:
            db_engine (sqlalchemy.engine)   : Database connection engine
            model_id (int)  : Model ID we are interested in
            k (Union[int, float]): If int, an absolute threshold will be considered. 
                                If float, the number has to be (0, 1] and a percentage threshold will be considered  

            fetch_all_ties (bool): Whether to include ties (can include more than k elements if k is int) 
                                    or randomly break ties

        return:
            a pd.DataFrame object that contains the entity_id, as_of_date, the hash of the text matrix, model score, 
            and all rank columns 
            
    """
    
    # Determining which rank column to filter by
    ties_suffix = 'no_ties'
    col = 'rank_pct'
    
    if fetch_all_ties:
        ties_suffix='with_ties'

    if isinstance(k, int):
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
        and {col_name} <= {k}
    """

    top_k = pd.read_sql(q, db_engine)

    return top_k
   


def compare_two_lists(list_df1, list_df2):
    """ Given two lists compare their similarities

        args:
            list_df1 (pd.DataFrame): A dataframe containing the top-k list for a model
            list_df2 (pd.DataFrame): A dataframe containing the top-k list for a model 
 
        return:
            similarities (dict): Dictionary that contains 'overlap', 'jaccard_similarity', 'rank_correlation'
    """

    if list_df1.as_of_date.at[0] != list_df2.as_of_date.at[0]:
        logging.warning('You are comparing two lists generated on different as_of_dates!')

    # calculating jaccard similarity and overlap
    entities_1 = set(list_df1.entity_id)
    entities_2 = set(list_df2.entity_id)

    inter = entities_1.intersection(entities_2)
    un = entities_1.union(entities_2)

    results = dict()
    
    results['jaccard'] = len(inter)/len(un)

    # If the list sizes are not equal, using the smallest list size to calculate simple overlap
    results['overlap'] = len(inter)/ min(len(entities_1), len(entities_2))


    # calculating rank correlation
    list_df1.sort_values('score', ascending=False, inplace=True)
    list_df2.sort_values('score', ascending=False, inplace=True)

    # only returning the corr coefficient, not the p-value
    results['rank_corr'] = spearmanr(list_df1.entity_id.iloc[:, 0], list_df2.entity_id.iloc[:, 1])[0]

    return results


def _get_crosstabs_highest_risk_vs_rest(model_id, matrix):
    """ generate crosstabs for the top_k vs the rest"""

    pass


def _get_descriptives(entities, columns_of_interest):
    """ Given a list of entities, generated descriptives

        args:
            entities (pd.DataFrame): The Dataframe containing information of the entities
            columns_of_interest (List[str]): The list of column names we are interested in describing
    """

    # TODO -- Not entirely sure how we can write this as a general function
    pass
