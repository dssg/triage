from collate import collate


class FeatureGenerator(object):
    def __init__(self, training_label_table, feature_aggregations, data_table):
        self.training_label_table = training_label_table
        self.data_table = data_table
        self.feature_aggregations = feature_aggregations

    def as_of_dates(self):
        # TODO: select from training labels table?
        return ['2016-04-01']

    def aggregation(self, aggregation_config):
        aggregates = [
            collate.Aggregate(aggregate['predicate'], aggregate['metrics'])
            for aggregate in aggregation_config['aggregates']
        ]
        group_intervals = {
            interval['name']: interval['intervals']
            for interval in aggregation_config['group_intervals']
        }
        return collate.SpacetimeAggregation(
            [aggregates],
            from_obj='{} ts join {} d on (ts.entity_id = d.entity_id)'.format(self.training_labels_table, self.data_table),
            group_intervals=group_intervals,
            dates=self.as_of_dates(),
            date_column='knowledge_date'
        )

    def generate(self):
        aggregations = [
            self.aggregation(aggregation_config)
            for aggregation_config in self.feature_aggregations
        ]
        # TODO: execute selects from this to a
        # features table and return the name of the table
        return 'features'
