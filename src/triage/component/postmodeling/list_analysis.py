""" This is a temporaty script for writing functions that analyse the top-k lists,
    while others are working on other components. Eventually these files will be merged
"""

import logging
import itertools
import pandas as pd

from scipy.stats import spearmanr


def get_highest_risk_k_entities(db_engine, model_id, k, include_all_tied_entities=False):
    """ Fetch the entities with the highest risk for a particular model
        
        args:
            db_engine (sqlalchemy.engine)   : Database connection engine
            model_id (int)  : Model ID we are interested in
            k (Union[int, float]): If int, an absolute threshold will be considered. 
                                If float, the number has to be (0, 1] and a percentage threshold will be considered  

            include_all_tied_entities (bool): Whether to include all entities with the same score in the list (can include more than k elements if k is int) 
                                    Defaults to False where ties are broken randomly

        return:
            a pd.DataFrame object that contains the entity_id, as_of_date, the hash of the text matrix, model score, 
            and all rank columns 
            
    """
    
    # Determining which rank column to filter by
    ties_suffix = 'no_ties'
    col = 'rank_pct'
    
    if include_all_tied_entities:
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
        risk_lists[model_id] = get_highest_risk_k_entities(db_engine, model_id, k, include_all_tied_entities)

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
