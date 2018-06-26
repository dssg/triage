import os
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import array, ARRAY
import pandas as pd
from triage.component.model_monitor.import_tool.sql_io import get_db_config, get_sqlalchemy_engine
from triage.component.model_monitor.mm_utils import get_mm_config


# tools for interfacing with a standard triage project


class TriageImportTool(object):
    def __init__(self,
                 db_config_path=None,
                 mm_config_path=None):

        self._db_config = get_db_config(db_config_path)
        self._engine = get_sqlalchemy_engine(self._db_config)

        if mm_config_path:
            self._mm_config = get_mm_config(mm_config_path)

            self.model_group_id_params = {
                'model_group_ids': self._mm_config['model_targets']['model_groups'],
                'no_model_group_subset': self._mm_config['model_targets']['no_model_group_subset']
            }

            self.model_id_params = {
                'model_ids': self._mm_config['model_targets']['model_ids'],
                'no_model_id_subset': self._mm_config['model_targets']['no_model_id_subset']
            }
        else:
            self._mm_config = dict()
            self.model_group_id_params = {
                'model_group_ids': [], 'no_model_group_subset': True
            }
            self.model_id_params = {
                'model_ids': [], 'no_model_id_subset': True
            }

    @staticmethod
    def _read_triage_query(fname):
        cdir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cdir, 'triage_default_queries', fname), mode='r') as f:
            return f.read()

    def select_predictions(self, start_date, end_date):

        # fetch query
        query = self._read_triage_query('predictions.sql')

        # construct params
        params = {'start_date': start_date,
                  'end_date': end_date}

        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query, con=self._engine, params=params)

    def select_feature_importances(self, start_date, end_date):
        # fetch query
        query = self._read_triage_query('feature_importances.sql')

        # construct params
        params = {'start_date': start_date,
                  'end_date': end_date}

        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query, con=self._engine, params=params)

    def select_models(self):
        # fetch query
        query = self._read_triage_query('models.sql')

        # construct params
        params = dict()
        params.update(self.model_group_id_params)
        params.update(self.model_id_params)

        # execute
        return pd.read_sql(query.format(**params), con=self._engine, params=params)

    def select_model_groups(self):
        # fetch query
        query = self._read_triage_query('model_groups.sql')

        # construct params
        params = self.model_group_id_params

        # execute
        return pd.read_sql(sa.text(query.format(**params)), con=self._engine)

    def select_prediction_lags_daily(self, end_date, compare_interval):
        # fetch query
        query = self._read_triage_query('get_prediction_lags_daily.sql')

        # construct params
        params = dict()
        params.update(self.model_group_id_params)
        params.update(self.model_id_params)
        params['compare_interval'] = compare_interval
        params['end_date'] = end_date

        # execute
        return pd.read_sql(sa.text(query.format(**params)), con=self._engine)

    def select_prediction_lags_hist(self, start_date, end_date, compare_interval):
        # fetch query
        query = self._read_triage_query('get_prediction_lags_hist.sql')

        # construct params
        params = dict()
        params.update(self.model_group_id_params)
        params.update(self.model_id_params)
        params['compare_interval'] = compare_interval
        params['start_date'] = start_date
        params['end_date'] = end_date

        # execute
        return pd.read_sql(sa.text(query.format(**params)), con=self._engine)

    def select_feature_importance_lags_daily(self, end_date, compare_interval):
        # fetch query
        query = self._read_triage_query('get_feature_importance_lags_daily.sql')

        # construct params
        params = dict()
        params.update(self.model_group_id_params)
        params.update(self.model_id_params)
        params['compare_interval'] = compare_interval
        params['end_date'] = end_date

        # execute
        return pd.read_sql(sa.text(query.format(**params)), con=self._engine)

    def select_feature_importance_lags_hist(self, start_date, end_date, compare_interval):
        # fetch query
        query = self._read_triage_query('get_feature_importance_lags_hist.sql')

        # construct params
        params = dict()
        params.update(self.model_group_id_params)
        params.update(self.model_id_params)
        params['compare_interval'] = compare_interval
        params['start_date'] = start_date
        params['end_date'] = end_date

        # execute
        return pd.read_sql(sa.text(query.format(**params)), con=self._engine)
