""" This is a temporaty script for writing functions that analyse the top-k lists,
    while others are working on other components. Eventually these files will be merged
"""

import pandas as pd


def get_highest_risk_k_entities(db_engine, model_id, k):
    """ Fetch the entities with the highest risk for a particular model
        
        args:
            db_engine (sqlalchemy.engine): Database connection engine
            model_id (int): Model ID we are interested in
            k (Union[int, float]): If int, an absolute threshold will be considered. 
                If float, the number has to be (0, 1] and a percentage threshold will be considered   
    """
    # TODO -- Specify tie breaking

    pass



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