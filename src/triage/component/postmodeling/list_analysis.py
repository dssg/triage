""" This is a temporaty script for writing functions that analyse the top-k lists,
    while others are working on other components. Eventually these files will be merged
"""

import pandas as pd


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
   



def compare_two_lists(list1, list2):
    """ Given two lists compare their similarities
        args:
            list1:
            list2:

        return:
            similarities (dict): Dictionary that contains 'overlap', 'jaccard_similarity', 'rank_correlation'
    """

    pass


def _get_descriptives(entities, columns_of_interest):
    """ Given a list of entities, generated descriptives

        args:
            entities (pd.DataFrame): The Dataframe containing information of the entities
            columns_of_interest (List[str]): The list of column names we are interested in describing
    """

    # TODO -- Not entirely sure how we can write this as a general function
    pass


def _get_crosstabs_highest_risk_vs_rest(predictions, matrix):
    """ generate crosstabs for the top_k vs the rest"""

    pass