import numpy as np
import pandas as pd
from triage.component.model_monitor.import_tool.sql_io import get_db_config, get_sqlalchemy_engine


PREDICTION_METRIC_DEF_KEYS = [
    'metric_type',
    'threshold',
    'use_top_entities',
    'use_lag_as_reference',
    'compare_interval'
]

PREDICTION_METRIC_KEYS = [
    'prediction_metric_id',
    'model_id',
    'as_of_date'
]

FEATURE_METRIC_DEF_KEYS = [
    'metric_type',
    'feature_block_agg',
    'threshold',
    'use_top_entities',
    'use_lag_as_reference',
    'compare_interval'
]

FEATURE_METRIC_KEYS = [
    'feature_metric_id',
    'model_id',
    'as_of_date'
]


class Loader(object):

    def __init__(self,
                 db_config_path=None):

        self._db_config = get_db_config(db_config_path)
        self._engine = get_sqlalchemy_engine(self._db_config)

    def push_prediction_metrics(self,
                                insert_df,
                                check_duplicate_metrics=False):
        assert isinstance(insert_df, pd.DataFrame), "Must provide pandas.DataFrame as input"

        # compare current calculated metric definitions to existing ones
        current_metrics = insert_df.drop_duplicates(subset=PREDICTION_METRIC_DEF_KEYS)

        existing_metrics = pd.read_sql("SELECT * FROM model_monitor.preidction_metric_defs",
                                       con=self._engine)

        max_metric_id = existing_metrics['prediction_metric_id'].max()

        merged_metrics = pd.merge(current_metrics,
                                  existing_metrics,
                                  how='left',
                                  on=PREDICTION_METRIC_DEF_KEYS)

        merged_metrics.loc[:, 'is_new_metric_id'] = merged_metrics['preidction_metric_id'].isnull()
        merged_metrics.reset_index(inplace=True)

        # if new metrics defined
        if len(merged_metrics[merged_metrics['is_new_metric_id']]) > 0:
            # assign new metric ids and push to SQL
            new_metric_defs = merged_metrics[merged_metrics['is_new_metric_id']]
            new_metric_defs.loc[:, 'prediction_metric_id'] = \
                pd.Series(range(len(new_metric_defs))) + max_metric_id + 1
            new_metric_defs[PREDICTION_METRIC_DEF_KEYS].to_sql(
                'model_monitor.prediction_metric_defs',
                con=self._engine,
                if_exists='append'
            )

            # add new metric ids to original table
            new_metric_id_map = dict(zip(new_metric_defs['index'], new_metric_defs['prediction_metric_id']))
            merged_metrics.loc[:, 'prediction_metric_id'] = np.where(
                merged_metrics['is_new_metric_id'],
                merged_metrics['prediction_metric_id'].apply(
                    lambda d: new_metric_id_map.get(d, default=None)
                ),
                merged_metrics['prediction_metric_id']
            )

        # compare current calculated metrics to existing ones (if necessary)
        if check_duplicate_metrics:
            # TODO: temp table SQL only solution, this is not good practice
            existing_metrics_values = pd.read_sql(
                "SELECT prediction_metric_id, model_id, as_of_date, 1 AS is_dup FROM model_monitor.prediction_metrics",
                con=self._engine
            )

            merged_metric_values = pd.merge(merged_metrics,
                                            existing_metrics_values,
                                            how='left',
                                            on=PREDICTION_METRIC_KEYS)

            merged_metrics = merged_metric_values[merged_metrics['is_dup'].isnull()]

        # push final results to SQL
        merged_metrics[PREDICTION_METRIC_KEYS].to_sql('model_monitor.prediction_metrics',
                                                      con=self._engine)

    def push_feature_metrics(self,
                             insert_df,
                             check_duplicate_metrics=False):
        assert isinstance(insert_df, pd.DataFrame), "Must provide pandas.DataFrame as input"

        # compare current calculated metric definitions to existing ones
        current_metrics = insert_df.drop_duplicates(subset=FEATURE_METRIC_DEF_KEYS)

        existing_metrics = pd.read_sql("SELECT * FROM model_monitor.feature_metric_defs",
                                       con=self._engine)

        max_metric_id = existing_metrics['feature_metric_id'].max()

        merged_metrics = pd.merge(current_metrics,
                                  existing_metrics,
                                  how='left',
                                  on=FEATURE_METRIC_DEF_KEYS)

        merged_metrics.loc[:, 'is_new_metric_id'] = merged_metrics['feature_metric_id'].isnull()
        merged_metrics.reset_index(inplace=True)

        # if new metrics defined
        if len(merged_metrics[merged_metrics['is_new_metric_id']]) > 0:
            # assign new metric ids and push to SQL
            new_metric_defs = merged_metrics[merged_metrics['is_new_metric_id']]
            new_metric_defs.loc[:, 'feature_metric_id'] = \
                pd.Series(range(len(new_metric_defs))) + max_metric_id + 1
            new_metric_defs[FEATURE_METRIC_DEF_KEYS].to_sql(
                'model_monitor.feature_metric_defs',
                con=self._engine,
                if_exists='append'
            )

            # add new metric ids to original table
            new_metric_id_map = dict(zip(new_metric_defs['index'], new_metric_defs['feature_metric_id']))
            merged_metrics.loc[:, 'feature_metric_id'] = np.where(
                merged_metrics['is_new_metric_id'],
                merged_metrics['feature_metric_id'].apply(
                    lambda d: new_metric_id_map.get(d, default=None)
                ),
                merged_metrics['feature_metric_id']
            )

        # compare current calculated metrics to existing ones (if necessary)
        if check_duplicate_metrics:
            # TODO: temp table SQL only solution, this is not good practice
            existing_metrics_values = pd.read_sql(
                "SELECT prediction_metric_id, model_id, as_of_date, 1 AS is_dup FROM model_monitor.feature_metrics",
                con=self._engine
            )

            merged_metric_values = pd.merge(merged_metrics,
                                            existing_metrics_values,
                                            how='left',
                                            on=FEATURE_METRIC_KEYS)

            merged_metrics = merged_metric_values[merged_metrics['is_dup'].isnull()]

        # push final results to SQL
        merged_metrics[PREDICTION_METRIC_KEYS].to_sql('model_monitor.feature_metrics',
                                                      con=self._engine)
