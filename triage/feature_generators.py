from collate.collate import Categorical, Aggregate
from collate.spacetime import SpacetimeAggregation
import logging


class FeatureGenerator(object):
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def aggregation(self, aggregation_config, feature_dates):
        aggregates = [
            Aggregate(aggregate['quantity'], aggregate['metrics'])
            for aggregate in aggregation_config.get('aggregates', [])
        ]
        logging.info('Found %s quantity aggregates', len(aggregates))
        categoricals = [
            Categorical(categorical['column'], categorical['choices'], categorical['metrics'])
            for categorical in aggregation_config.get('categoricals', [])
        ]
        logging.info('Found %s categorical aggregates', len(categoricals))
        return SpacetimeAggregation(
            aggregates + categoricals,
            from_obj=aggregation_config['from_obj'],
            intervals=aggregation_config['intervals'],
            groups=aggregation_config['groups'],
            dates=feature_dates,
            date_column=aggregation_config['knowledge_date_column'],
            prefix=aggregation_config['prefix']
        )

    def generate(self, feature_aggregations, feature_dates):
        aggregations = [
            self.aggregation(aggregation_config, feature_dates)
            for aggregation_config in feature_aggregations
        ]
        for aggregation in aggregations:
            logging.warning('getting selects')
            for selectlist in aggregation.get_selects().values():
                for select in selectlist:
                    logging.warning(select)
            aggregation.execute(self.db_engine.connect())
        return [agg.get_table_name() for agg in aggregations]
