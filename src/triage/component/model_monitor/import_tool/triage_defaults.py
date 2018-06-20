import os
import sqlalchemy as sa
import pandas as pd
from .sql_io import get_db_config, get_sqlalchemy_engine
from ..mm_utils import get_mm_config


# tools for interfacing with a standard triage project


class TriageImportTool(object):
    def __init__(self,
                 db_config=None):
        self._db_config = db_config if db_config else get_db_config()
        self._engine = get_sqlalchemy_engine(self._db_config)
        self._mm_config = get_mm_config()

        self.model_group_id_params = {
            'model_groups': self._mm_config['model_targets']['model_groups'],
            'no_model_group_subset': self._mm_config['model_targets']['no_model_group_subset']
        }

        self.model_id_params = {
            'model_ids': self._mm_config['model_targets']['model_ids'],
            'no_model_id_subset': self._mm_config['model_targets']['no_model_id_subset']
        }

    @staticmethod
    def _read_triage_query(fname):
        with open(os.path.join('triage_default_queries', fname), mode='r') as f:
            return sa.text(f.read())

    def select_predictions(self, start_date, end_date):

        # fetch query
        query = self._read_triage_query('predictions')

        # construct params
        params = {'start_date': start_date,
                  'end_date': end_date}

        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query, con=self._engine, params=params)

    def select_feature_importances(self, start_date, end_date):
        # fetch query
        query = self._read_triage_query('feature_importances')

        # construct params
        params = {'start_date': start_date,
                  'end_date': end_date}

        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query, con=self._engine, params=params)

    def select_models(self):
        # fetch query
        query = self._read_triage_query('predictions')

        # construct params
        params = dict()

        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query, con=self._engine, params=params)

    def select_model_groups(self):
        # fetch query
        query = self._read_triage_query('predictions')

        # construct params
        params = self.model_group_id_params

        # execute
        return pd.read_sql(query, con=self._engine, params=params)
