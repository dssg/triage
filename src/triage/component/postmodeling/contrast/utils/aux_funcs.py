'''
Auxiliary functions and helpers:

This set of functions are helper functions to format data 
(i.e., prediction matrices, etc.) for plotting. This functions
are called in both Model class and ModelGroup class in 
evaluation.py.
'''

from sqlalchemy import create_engine
from sqlalchemy.sql import text
from collections import namedtuple
import pandas as pd
import yaml

def create_pgconn(credentials_yaml):
    '''
    Create SQL connection object using a psycopg2 cursor and abiding to new
    dssg/dsapp db user configuration. 

    Arguments:
        - credentials_yaml: .yaml file with db credentials
    '''
    with open(credentials_yaml) as f:
        configs = yaml.load(f)
    try:
        conn = create_engine("postgresql://{user}:{password}@{host}/{dbname}".format(**configs))
    except:
        print("Error connecting to db.")

    return conn 

def read_parameters(postmodeling_parameters):
    '''
    Read YAML file with post-modeling parameters. 
    '''
    with open(postmodeling_parameters, 'r') as f:
        postmodeling_config = yaml.load(f)
        for key, val in postmodeling_config['metrics'].items():
            exec(key + '=val')

def get_models_ids(audited_model_group_ids, conn):
    '''
    This helper functions will retrieve the model_id's from a set
    of model_group_ids and will instantiate each model into the 
    ModelEvaluator class. 

    Aguments:
        - audited_model_group_ids: List of model_group_ids 
          (ideally from Audition's output)
        - conn: sql engine

    This function will return a list of ModelEvaluator objects
    '''

    query = conn.execute(text("""
    SELECT model_group_id,
           model_id
    FROM model_metadata.models
    WHERE model_group_id = ANY(:ids);
    """), ids=audited_model_group_ids)

    ModelTuple = namedtuple('ModelEvaluator', ['model_group_id', 'model_id'])
    l = [ModelTuple(*i) for i in query]

    return(l)

def recombine_categorical(df_dest, df_source, prefix, suffix='', entity_col='entity_id'):
    """Combine a categorical variable that has been one-hot encoded into binary columns
       back into a single categorical variable (assumes the naming convention of collate).

       The binary columns here are assumed to be mutually-exclusive and each entity will
       only be given (at most) a single categorical value (in practice, this will be the
       last such value encountered by the for loop).

       Note: modifies the input data frame, df_dest, directly.

    Arguments:
        df_dest (DataFrame) -- data frame into which the recombined categorical variable will be stored
        df_source (DataFrame) -- data frame with the one-hot encoded source columns
        prefix (string) -- prefix shared by the binary columns for the categorical variable, typically
                           something like 'feature_group_entity_id_1y_columnname_'
        suffix (string) -- suffix shared by the binary columns for the categorical variable, typically
                           something like '_min', '_max', '_avg', etc.
        entity_col (string) -- column to identify entities to map the 0/1 binary values in the one-hot
                               columns to catagorical values in the recombined column.

    Returns: tuple of the modified df_dest and the new categorical column name
    """
    cat_col = prefix+suffix+'_CATEGORIES'
    df_dest[cat_col] = np.NaN
    df_dest[cat_col] = df_dest[cat_col].astype('category')

    for col in df_source.columns[df_source.columns.str.startswith(prefix) & df_source.columns.str.endswith(suffix)]:
        cat_val = col.replace(prefix, '').replace(suffix, '')
        df_dest[cat_col].cat.add_categories([cat_val], inplace=True)
        cat_entities = df_source.loc[df_source[col]==1][entity_col]
        df_dest.loc[df_dest[entity_col].isin(cat_entities), cat_col] = cat_val

        return (df_dest, cat_col)

if __name__ == "utils.aux_funcs":
    conn = create_pgconn('db_credentials.yaml')


