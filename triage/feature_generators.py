from collate.collate import collate
import logging


class FeatureGenerator(object):
    def __init__(self, feature_aggregations, feature_dates, data_table, db_engine):
        self.data_table = data_table
        self.feature_dates = feature_dates
        self.feature_aggregations = feature_aggregations
        self.db_engine = db_engine

    def aggregation(self, aggregation_config):
        aggregates = [
            collate.Aggregate({aggregate['name']: aggregate['predicate']}, aggregate['metrics'])
            for aggregate in aggregation_config['aggregates']
        ]
        group_intervals = {
            interval['name']: interval['intervals']
            for interval in aggregation_config['group_intervals']
        }
        return collate.SpacetimeAggregation(
            aggregates,
            from_obj=self.data_table,
            group_intervals=group_intervals,
            dates=self.feature_dates,
            date_column='knowledge_date',
            prefix=aggregation_config['prefix']
        )

    def _table_names(self):
        all_tables = []
        for aggregation_config in self.feature_aggregations:
            for group_interval in aggregation_config['group_intervals']:
                all_tables.append('{}_{}'.format(aggregation_config['prefix'], group_interval['name']))
        return all_tables

    def generate(self):
        aggregations = [
            self.aggregation(aggregation_config)
            for aggregation_config in self.feature_aggregations
        ]
        for aggregation in aggregations:
            aggregation.execute(self.db_engine.connect())
        return self._table_names()
