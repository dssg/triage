import pandas as pd 
import numpy as np

from scipy.stats import gaussian_kde


def get_evaluations_for_metric(db_engine, model_group_ids, metric, parameter=None, subset_hash=''): 
    """ 
        Retrieves the evaluations associated with a list of model group ids, metric and parameter of interest. 

        Args: 
            db_engine(sqlalchemy): Database engine to connect to the DB
            model_group_ids (list): List of model_group_id
            metric (str): Performance metric of interest
            parameter (str, optional): Threshold of interest. If not provided, metric values for all the parameters are returned
            subset_hash (str, optional): Hash of the cohort subset. Defaults to an empty string (whole cohort)

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
        and subset_hash = '{subset_hash}'
    """
    
    if parameter is not None: 
        q += f"and parameter = '{parameter}'"

    evaluations = pd.read_sql(q, db_engine)

    return evaluations


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


def generate_overall_kde(scores_df):
    """Generates the Kernel Density Estimation for each model type """
    kde_list_model_types = []

    # Loop over each model type
    #for model, df in scores.groupby("model_type_display"):
    for model_type in scores_df.model_type_display.unique():
        df = scores_df[scores_df.model_type_display == model_type]
        kde = gaussian_kde(df.score.values)
        
        # Define x values for the density curve
        x_vals = np.linspace(0, 1, 200)
        y_vals = kde(x_vals)
        
        # Save to dataframe
        kde_df = pd.DataFrame({
            "score": x_vals,
            "density": y_vals,
            "model_type_display": model_type
        })
        
        kde_list_model_types.append(kde_df)

    overall_density_df = pd.concat(kde_list_model_types, ignore_index=True)

    return overall_density_df


def generate_kde_by_model_timechop(scores_df):
    """Calculates the Kernel Density Estimation for a timechop-model type"""
    kde_list_model_timechop = []

    # Loop over each model type
    #for model, df in scores.groupby("model_type_display"):
    for timechop in scores_df.train_end_time.unique():
        for model_type in scores_df.model_type_display.unique():
            df = scores_df[(scores_df.model_type_display == model_type) & 
                        (scores_df.train_end_time == timechop)]
            kde = gaussian_kde(df.score.values)
            
            # Define x values for the density curve
            x_vals = np.linspace(0, 1, 200)
            y_vals = kde(x_vals)
            
            # Save to dataframe
            kde_df = pd.DataFrame({
                "score": x_vals,
                "density": y_vals,
                "train_end_time": timechop,
                "model_type_display": model_type
            })
            
            kde_list_model_timechop.append(kde_df)

    density_timechop_df = pd.concat(kde_list_model_timechop, ignore_index=True)

    return density_timechop_df