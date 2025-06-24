import pandas as pd 

from itertools import combinations


def get_evaluations_for_metric(model_group_ids, metric, parameter, db_engine): 
    """ 
        Retrieves the evaluations associated with a list of model group ids, metric and parameter of interest. 

        Args: 
            model_group_ids (list): List of model_group_id
            metric (str): Performance metric of interest
            parameter (str): Threshold of interest

        Returns:
            pandas.DataFrame with the evaluation value associated with the metric and threshold defined for the list of models sent 
    """
    model_groups_sql = str(model_group_ids).replace('[','').replace(']','')
    #parameters_sql = str(parameters).replace('[','').replace(']','')

    q = f"""
        select
            model_group_id,
            model_id, 
            model_type,
            hyperparameters,
            evaluation_end_time as as_of_date,
            metric, 
            parameter,
            stochastic_value as value         
        from triage_metadata.models a
        join test_results.evaluations b
        using (model_id) 
        where model_group_id in ({model_groups_sql})
        and metric = '{metric}'
        and parameter = '{parameter}'
    """

    evaluations_for_model_id = pd.read_sql(q, db_engine)

    return evaluations_for_model_id


def get_evaluations_from_model_group(model_group_ids, metrics, parameters, db_engine):
    """
        Gets the the evaluations for specified metrics and parameters generated for specified model groups

        Args 
            model_group_ids (list): A list of model group ids of interest
            metrics (list): A list of metrics of interes
            parameters (set): A set of parameters of interest

        Returns
            evaluations (pd.DataFrame): A pandas data frame with the specified evaluations for the model groups of interest 
    
    """
    model_groups_sql = str(model_group_ids).replace('[','').replace(']','')
    metrics_sql = str(list(metrics)).replace('[','').replace(']','')
    parameters_sql = str(parameters).replace('[','').replace(']','')

    q = f"""
        select
            model_group_id,
            model_id, 
            model_type,
            hyperparameters,
            evaluation_end_time as as_of_date,
            metric, 
            parameter,
            stochastic_value as value         
        from triage_metadata.models a
        join test_results.evaluations b
        using (model_id) 
        where model_group_id in ({model_groups_sql})
        and metric in ({metrics_sql})
        and parameter in ({parameters_sql})
    """

    evaluations_for_model_id = pd.read_sql(q, db_engine)

    return evaluations_for_model_id


def get_pairs_models_groups_comparison(model_groups_ids):
    """Given a list of model group ids, generates a list of all possible pairs"""
    pairs = list(combinations(model_groups_ids, 2))
    
    return pairs


def validation_group_model_exists(model_group_id, db_engine):
    """Verifies if the model group exist"""
    q = f"""
        select distinct 
            model_id
        from triage_metadata.models 
        where model_group_id = {model_group_id}
    """

    models_in_model_group = pd.read_sql(q, db_engine)

    if models_in_model_group.shape[0] > 0: 
        return True
    else:
        return False
    

def validation_metric_generated(model_group_id, metrics, db_engine):
    """Verifies if the metric was generated for the model group id"""
    metrics_sql = str(metrics).replace("[", "").replace("]", "")

    q = f"""
        select distinct
            model_id, 
            metric
        from triage_metadata.models a 
        join test_results.evaluations b
        using (model_id)
        where model_group_id = {model_group_id}
        and metric in ({metrics_sql})
    """ 

    metrics_evaluated = pd.read_sql(q, db_engine)

    return metrics_evaluated
    