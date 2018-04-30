""" Create and populate the results.crosstabs table. This table provides 
information on how low-risk and high-risk entities differ in terms of their
features.

Note that this module is using thresholds and get_prediction_query from the
bias module; these might need to be swapped out when you call 
populate_crosstabes_table().
"""

import pandas as pd
import os
import json
import logging
from scipy import stats
from sqlalchemy import create_engine

#from postmodel.utils import get_engine
#from postmodel.bias.bias import get_predictions_query, thresholds


#########################################
#     Default Crosstab Functions        #
#########################################

# Later to be run against feature matrices of high- and low-risk entities.
# Each function receives two arguments: A dataframe of high-risk entities'
# features (as columns), and a dataframe of low-risk entities' features.
# The functions return either a Series, where the row index are the names of 
# the feature columns, or a dataframe, again with the rows indexed by feature
# names. 

high_risk_n = lambda hr, lr: pd.Series(index=hr.columns, data=hr.shape[0])
low_risk_n = lambda hr, lr: pd.Series(index=lr.columns, data=lr.shape[0])
high_risk_mean = lambda hr, lr: hr.mean(axis=0)
low_risk_mean = lambda hr, lr: lr.mean(axis=0)
high_risk_std = lambda hr, lr: hr.std(axis=0)
low_risk_std = lambda hr, lr: lr.std(axis=0)
hr_lr_ratio = lambda hr, lr: hr.mean(axis=0)/lr.mean(axis=0)

def get_engine():

    with open(os.path.join(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__))), 'database_credentials.json'), 'r') as f:
       
        creds = json.load(f)
    
    conn_str = "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
            **creds)

    return create_engine(conn_str)


def hr_lr_ttest(hr, lr):
    """ Returns the t-test (T statistic and p value), comparing the features for
    high- and low-risk entities. """

    res = stats.ttest_ind(hr.as_matrix(), lr.as_matrix(), axis=0, nan_policy='omit',
                          equal_var=False)

    r0 = pd.Series(res[0], index=hr.columns)
    r1 = pd.Series(res[1], index=hr.columns)

    return pd.DataFrame({'ttest_T': r0, 'ttest_p': r1})

# the crosstab functions are passed as a list of tuples;
# the first tuple element gives the name (or names) of the 
# metric that the function calculates
crosstab_functions = [
         ('count_predicted_positive', high_risk_n),
         ('count_predicted_negative', low_risk_n),
         ('mean_predicted_positive', high_risk_mean),
         ('mean_predicted_negative', low_risk_mean),
         ('std_predicted_positive', high_risk_std),
         ('std_predicted_negative', low_risk_std),
         ('ratio_predicted_positive_over_predicted_negative', hr_lr_ratio),
         (['ttest_T','ttest_p'], hr_lr_ttest)
        ]

#######################################
#######################################

def create_crosstabs_table(engine=None, schema='results', table='crosstabs'):

    if not engine:
        engine = get_engine()

    with open(os.path.join(os.path.dirname(__file__), 
                           'sql/create_crosstabs.sql'),'r') as f:
        engine.execute(f.read().format(schema=schema, table=table))

def list_to_str(list_sql):
    str = '(' + ','.join(list_sql) + ')'
    return str


def get_predictions_query(model_id, as_of_date, entity_id_list):
    
    prediction_query = '''
    SELECT 
        model_id,
        as_of_date,
        entity_id,
        score,
        label_value,
        coalesce(rank_abs,rank() over (order by score desc)) as rank_abs,
        coalesce(rank_pct*100, ntile(100) over (order by score desc)) as rank_pct
    FROM results.predictions
    WHERE model_id={model_id} 
    AND as_of_date='{as_of_date}'
    '''.format(model_id=model_id, 
               as_of_date=as_of_date)
   
    return prediction_query


def get_sfpd_baseline_predictions_query(model_id, as_of_date, entity_id_list):

    prediction_query = '''
    SELECT
        model_id,
        as_of_date,
        entity_id,
        score,
        label_value,
        rank() over (order by score desc) as rank_abs,
        ntile(100) over (order by score desc) as rank_pct
    FROM results.eis_predictions
    WHERE model_id={model_id}
    AND as_of_date='{as_of_date}'
    '''.format(model_id=model_id,
               as_of_date=as_of_date)

    return prediction_query


def get_features_query(model_id, as_of_date, entity_id_list):
    """ Returns a query with column model_id, as_of_date, entity_id,
        plus an arbitrary number of other columns that list features.

        The table should be unique in entity_id, and model_id and as_of_date
        should be identical for all rows.

        If entity_id_list is provided, then the resulting table should be 
        restricted to members of that list.

        Note that this query might (or might not) be independent of model_id; 
        the resulting table still should return the model_id as a column, 
        which always should be constant across rows.
 
        This function needs to be implemented for each project. An example
        implementation for the SFPD pipeline is provided below.

        Args:
            model_id (int): a model ID
            as_of_date (str): an as_of_date
            entity_id_list ([int]): list of entity IDs; only members
                                    of this lists should have rows in the
                                    resulting dataframe
        Returns (str): A SQL query, giving columns model_id, as_of_date,
                       entity_id, plus an arbitrary amount of feature columns.
    """
    raise NotImplementedError


def get_sfpd_features_query(model_id, as_of_date, entity_id_list):
    """ This is an example implementation of get_features_query() for the
    SFPD project. Here, we need to restrict the number of features we fetch,
    because Postgres only support ~1600 columns. """

    """query = '''
        select {model_id}::INT as model_id, *
        from features.arrests_entity_id f1
        full outer join features.compliments_entity_id f2 using (entity_id, as_of_date)
        full outer join features.dispatches_entity_id f3 using (entity_id, as_of_date)
        full outer join features.incidents_entity_id f4 using (entity_id, as_of_date)
        -- full features.trafficstops_entity_id f5 using (entity_id, as_of_date)
        -- full outer join features.training_entity_id f6 using (entity_id, as_of_date)
        -- full outer join features.useofforce_entity_id f7 using (entity_id, as_of_date)
        where as_of_date='{as_of_date}'
    '''.format(model_id=model_id, as_of_date=as_of_date)
"""

    query = '''
        select {model_id}::INT as model_id, *
        from features.events_aggregation_imputed f1
        --full outer join features.compliments_entity_id f2 using (entity_id, as_of_date)
        --full outer join features.dispatches_entity_id f3 using (entity_id, as_of_date)
        --full outer join features.incidents_entity_id f4 using (entity_id, as_of_date)
        -- full features.trafficstops_entity_id f5 using (entity_id, as_of_date)
        -- full outer join features.training_entity_id f6 using (entity_id, as_of_date)
        -- full outer join features.useofforce_entity_id f7 using (entity_id, as_of_date)
        where as_of_date='{as_of_date}'
    '''.format(model_id=model_id, as_of_date=as_of_date)


    if entity_id_list:
        query += " and entity_id=ANY('{%s}') "%', '.join(map(str, entity_id_list))

    return query

def get_la_features_query(model_id, as_of_date, entity_id_list):
    """ This is an example implementation of get_features_query() for the
    SFPD project. Here, we need to restrict the number of features we fetch,
    because Postgres only support ~1600 columns. """

    query = '''
        select {model_id}::INT as model_id, *
        from features.demos_entity_id f1
        full outer join features.num_prior_entity_id f2 using (entity_id, as_of_date)
        where as_of_date='{as_of_date}'
    '''.format(model_id=model_id, as_of_date=as_of_date)

    if entity_id_list:
        query += " and entity_id=ANY('{%s}') "%', '.join(map(str, entity_id_list))

    return query



def populate_crosstabs_table(model_id, as_of_date,
                             get_features_query=None,
                             get_features_df=None,
                             get_predictions_query=None,
                             entity_id_list=None,
                             thresholds=None,
                             crosstab_functions=crosstab_functions,
                             push_to_db=True,
                             return_df=False,
                             engine=None,
                             schema='results',
                             table='crosstabs'): 
    """ This function populates the results.crosstabs table.
        
        Args:
            models_of_interest ([()]): A list of tuples with (model_id, as_of_date).
                                       This function will populate the crosstab 
                                       table for every (model_id, as_of_date) 
                                       in this list.
            get_features_query (func): A function that returns a SQL query (str).
                                       For its expected output and arguments, see the 
                                       example implementation further up in 
                                       this module.
                                       If your features are not available from your 
                                       Postgres DB, you can instead provide a
                                       dataframe-returning function via the 
                                       get_features_df argument. In that case,
                                       get_features_query needs to be None.
            get_features_df (func): A function that model_id, as_of_date, and 
                                    entity_id_list and returns a Pandas DataFrame.
                                    The DataFrame needs to be indexed by model_id, 
                                    as_of_date, and entity_id, and have an arbitrary 
                                    number of features as columns. The dataframe
                                    should be unique in entity_id, and model_id 
                                    and as_of_date should be identical for all rows.
                                    If entity_id_list is provided, then the 
                                    resulting dataframe should be restricted to 
                                    members of that list.
                                    Alternatively, you can supply a function that
                                    returns a Postgres-query via the 
                                    get_features_query argument. In that case,
                                    get_features_df should be None.
            get_predictions_query (func): A function that returns a SQL query to fetch
                                          predictions. For its expected input and output,
                                          see the example implementation in the bias module.
            entity_id_list (list): A list of entity_ids. Crosstabs will 
                                   be calculated using only predictions and 
                                   features of these entities. This list is
                                   being passed to get_predictions_query and
                                   get_features_query.
            crosstab_functions (dict): A dictionary of function names and 
                                   functions. The functions receive the feature
                                   dataframes for entities that were predicted 
                                   postive or negative, respectively. Each function
                                   returns a Series or DataFrame with rows now
                                   indexed by the feature names, and columns 
                                   corresponding to the statistics that the
                                   function calculated.
            thresholds (dict): A dictionary that maps column names to lists of 
                               threshold values. The (name, threshold) pairs will
                               be applied to the dataframes with predictions to 
                               split low- and high-risk entities.
            push_to_db (bool): If True, the crosstab table is being populated.
            return_df (bool): If True, a dataframe corresponding to the crosstabe 
                              table will be returned.
            engine (SQLAlchemy engine): DB connection
            schema (str): Name of the schema to store results; defaults to 'results'
            table (str): Name of table to store results; defaults to 'crosstabs'

            Returns: A DataFrame, corresponding to results.crosstabs
    """

    if get_features_query==None and get_features_df==None:
        raise ValueError("You need to supply features either as a Postgres "
                "query or as a Pandas DataFrame.")

    if get_features_query!=None and get_features_df!=None:
        raise ValueError("You need to supply features as a Postgres query "
                "or as a Pandas DataFrame, but not both.")

    if not engine:
        engine = get_engine()

    dfs = []
    
    # this will be useful later - the non-feature columns we'll grab
    model_cols = ['model_id','as_of_date','entity_id','score','rank_abs',
                  'rank_pct','label_value']

    # if features are provided as a Postgres query, we can fetch the table
    if get_features_query:
        query = '''
            with prediction_table as (
            {predictions_query}
            ), feature_data as (
            {features_query}
            )
            SELECT * FROM prediction_table 
            LEFT JOIN feature_data USING (model_id,entity_id, as_of_date);    
        '''.format(predictions_query=pquery,
                   features_query=fquery)
    
        # grab the data
        df = pd.read_sql(query, engine) 

    # otherwise, the features must be presented as a dataframe
    elif get_features_df:
        query = get_predictions_query(model_id, as_of_date, entity_id_list)
        predictions_df = pd.read_sql(query, engine).set_index(['model_id','as_of_date','entity_id'])

        df = predictions_df.join(get_features_df(model_id, as_of_date, 
                                                  entity_id_list),
                                  how='left')

    if len(df)==0:
        raise ValueError("No data could be fetched.")

    # drop the all-null columns
    null_cols = df.columns[df.isnull().sum()==df.shape[0]]
    null_cols = [c for c in null_cols if c not in model_cols]

    if len(null_cols)>0:
        logging.warning("Dropping the all-null columns: %s"%str(null_cols))
        df = df.drop(null_cols, 1)

    # if the dataframe isn't dummified, do it
    if object in df.dtypes.values or pd.Categorical.dtype in df.dtypes.values:
        to_dummify = [c for c in df.columns 
                      if pd.Categorical.dtype==df[c].dtype or df[c].dtype==object
                      and c not in model_cols]
        logging.info("Dummifying the data")
        df = pd.get_dummies(df, columns=to_dummify, dummy_na=True)

    feat_cols = [c for c in df.columns if c not in model_cols]

    for thres_type, thress in thresholds.items():
        for thres in thress:
            
            results = pd.DataFrame(index=feat_cols)
           
            # split dataframe into high/low risk
            df_pred_pos = df.loc[df[thres_type]<=thres,feat_cols]
            df_pred_neg = df.loc[df[thres_type]>thres,feat_cols]

            for name, func in crosstab_functions:

                # gets a Series or df, indexed by feature column names, with the
                # results of this function as columns

                this_result = pd.DataFrame(func(df_pred_pos, df_pred_neg))
                if name in ['ttest_T','ttest_p']:
                   print("this_result:", this_result.shape)
                   print("Results:", results.shape)
                if not type(name) in [list, tuple]:
                    name = [name]

                # the metric name is coming from the crosstab_functions tuples
                this_result.columns = name
                results = results.join(this_result,
                                       how='outer')

            results = results.stack()
            results.index.names = ['feature_column','metric']
            results.name = 'value'
            results = results.to_frame()

            results['model_id'] = model_id
            results['as_of_date'] = as_of_date
            results['threshold_unit'] = thres_type[-3:]
            results['threshold_value'] = thres

            dfs.append(results)

    df = pd.concat(dfs)

    if push_to_db:
        df.reset_index().set_index(['model_id','as_of_date','metric'])\
          .to_sql(schema=schema, name=table, 
                  con=engine, if_exists='append')

    if return_df:
        return df




if __name__ == '__main__':
        # new sfpd run
	# models_id = ['761', '795', '829', '863', '897']
	# dates_interest = ['2016-01-01', '2016-04-01', '2016-07-01', '2016-10-01', '2017-01-01']
	# sfpd baseline model run

  models = 'dsapp_eis'

  if (models == 'baseline_eis'):
    models_id = ['-1','-2','-3','-4']
    dates_interest = ['2016-01-01', '2016-04-01', '2016-07-01', '2016-10-01']
    thresholds_project =  {'rank_abs': [77,148,165,203]}

  elif (models == 'dsapp_eis'):
    models_id = ['761', '795', '829', '863', '897']
    dates_interest = ['2016-01-01', '2016-04-01', '2016-07-01', '2016-10-01', '2017-01-01']
    thresholds_project =  {'rank_abs': [77,148,165,203]}

  else:
    print ("error")

  models_of_interest = zip(models_id, dates_interest)

  '''
	Now we'll loop over the selected/listed models to populate the 
	crosstabs SQL table
  '''	
  for model_id, date_interest in models_of_interest:

    fquery = get_sfpd_features_query(model_id, date_interest ,0)
    if (models == 'baseline_eis'):
      pquery = get_sfpd_baseline_predictions_query(model_id, date_interest,0)
    elif (models == 'dsapp_eis'):
      pquery = get_predictions_query(model_id, date_interest,0)
    else:
      print ("error")



    a = populate_crosstabs_table(model_id=model_id, as_of_date=date_interest,
		  get_features_query=fquery,
		  get_features_df=None,
		  get_predictions_query=pquery,
		  entity_id_list=None,
		  thresholds=thresholds_project,
		  crosstab_functions=crosstab_functions,
		  push_to_db=True,
		  return_df=False,
		  engine=None,
		  schema='results',
		  table='crosstabs')