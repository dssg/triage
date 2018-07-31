"""
Model Evaluator

This script contain a set of elements to help the postmodeling evaluation
of audited models by triage.Audition. This will be a continuing list of
routines that can be scaled up and grow according to the needs of the
project, or other postmodeling approaches.

To run most of this routines you will need:
    - S3 credentials (if used) or specify the path to both feature and
    prediction matrices.
    - Working database db_engine.
"""

import pandas as pd
from sqlalchemy.sql import text
from collections import namedtuple
from utils.file_helpers import download_s3
from utils.test_conn import db_engine

# Get individual model_ids from Audition outcome


def get_models_ids(audited_model_group_ids):
    exec=db_engine.execute(text("""
    SELECT model_group_id,
           model_id
    FROM model_metadata.models
    WHERE model_group_id = ANY(:ids);
    """), ids=audited_model_group_ids)

    l = [i for i in exec]

    return(l)

# Get indivual model information/metadata from Audition output

class ModelExtractor(object):
    '''
    ModelExtractor class calls the model metadata from the database
    and hold model_id metadata features on each of the class attibutes.
    This class will contain any information about the model, and will be
    used to make comparisons and calculations across models. 

    A pair of (model_group_id, model_id) is needed to instate the class. These
    can be feeded from the get models_ids. 
    '''
    def __init__(self, model_group_id, model_id):
        self.model_id = model_id
        self.model_group_id = model_group_id

        # Retrive model_id metadata from the model_metadata schema
        model_metadata = pd.read_sql("""
        WITH
        individual_model_ids_metadata AS(
        SELECT m.model_id,
               m.model_group_id,
               m.hyperparameters,
               m.model_hash,
               m.experiment_hash,
               m.train_end_time,
               m.train_matrix_uuid,
               m.training_label_timespan,
               mg.model_type,
               mg.model_config
            FROM model_metadata.models m
            JOIN model_metadata.model_groups mg
            USING (model_group_id)
            WHERE model_group_id = {}
            AND model_id = {}
        ),
        individual_model_id_matrices AS(
        SELECT DISTINCT ON (matrix_uuid)
               model_id,
               matrix_uuid
            FROM test_results.predictions
            WHERE model_id = ANY(
                SELECT model_id
                FROM individual_model_ids_metadata
            )
        )
        SELECT metadata.*, test.*
        FROM individual_model_ids_metadata AS metadata
        LEFT JOIN individual_model_id_matrices AS test
        USING(model_id);
        """.format(self.model_group_id, self.model_id),
                   con=db_engine)
     
        # Add metadata attributes to model
        self.hyperparameters = model_metadata.loc[0, 'hyperparameters']
        self.model_hash = model_metadata.loc[0, 'model_hash']
        self.train_matrix_uuid = model_metadata.loc[0, 'train_matrix_uuid']
        self.matrix_uuid = model_metadata.loc[0, 'matrix_uuid']

   # def load_matrices(self, path):

   #    if 's3' in path:
   #        self.train_matrix = download_s3(str(path + self.train_matrix_uuid))
   #        self.test_matrix = download_s3(str(path + self.matrix_uuid))
   #    else:
   #        self.train_matrix = pd.read_csv(str(path + self.train_matrix_uuid + '.csv'))
   #        self.test_matrix = pd.read_csv(str(path + self.matrix_uuid + '.csv'))

