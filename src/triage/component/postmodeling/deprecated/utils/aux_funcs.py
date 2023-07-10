'''
A4uxiliary functions and helpers:

This set of functions are helper functions to format data
(i.e., prediction matrices, etc.) for plotting. This functions
are called in both Model class and ModelGroup class in
evaluation.py.
'''

from sqlalchemy import create_engine
from sqlalchemy.sql import text
from collections import namedtuple
import yaml

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)


ModelEvaluator = namedtuple('ModelEvaluator',
                           ('model_group_id', 'model_id'))


def create_pgconn(credentials_yaml):
    '''
    Create SQL connection object using a psycopg2 cursor and abiding to new
    dssg/dsapp db user configuration.

    Arguments:
        - credentials_yaml: .yaml file with db credentials
    '''
    with open(credentials_yaml) as f:
        configs = yaml.full_load(f)
    try:
        conn = create_engine("postgresql://{user}:{password}@{host}:{port}/{dbname}".format(**configs))
    except:
        logger.error("Error connecting to db.")

    return conn


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
    FROM triage_metadata.models
    WHERE model_group_id = ANY(:ids);
    """), ids=audited_model_group_ids)

    return [ModelEvaluator._make(row) for row in query]
