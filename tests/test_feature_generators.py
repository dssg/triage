#from triage.feature_generators import FeatureGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import logging
import pytest


input_data = [
    # entity_id, knowledge_date, category, quantity_one, quantity_two
    (1, date(2014, 1, 1), 'a thing', 10000, None),
    (1, date(2014, 10, 11), 'a thing', None, 40404),
    (3, date(2012, 6, 8), 'a thing', 342, 9234),
    (3, date(2014, 12, 21), 'another thing', 600, None),
    (4, date(2014, 4, 4), 'another thing', 1236, 6270)
]

aggregate_config = [{
    'prefix': 'aprefix',
    'aggregates': [
        { 'name': 'q1', 'predicate': 'quantity_one', 'metrics': ['sum', 'count'] },
        { 'name': 'q2', 'predicate': 'quantity_two', 'metrics': ['count', 'min'] },
    ],
    'group_intervals': [
        { 'name': 'entity_id', 'intervals': ['all'] },
    ]
}]

expected_output = {
    'aprefix_entity_id': [
        # entity_id, date, quantity_one_sum, quantity_one_count, quantity_two_count, quantity_two_min
        (3, date(2013, 9, 30), 342, 1, 1, 9234),
        (1, date(2014, 9, 30), 10000, 1, 0, None),
        (3, date(2014, 9, 30), 342, 1, 1, 9234),
        (4, date(2014, 9, 30), 1236, 1, 1, 6270),
    ]

}

@pytest.mark.skip(reason='collate package currently broken')
def test_training_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())

        engine.execute(
            'create table data (entity_id int, knowledge_date date, category varchar, quantity_one float, quantity_two float)'
        )
        for row in input_data:
            engine.execute(
                'insert into data values (%s, %s, %s, %s, %s)',
                row
            )

        output_tables = FeatureGenerator(
            data_table='data',
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregations=aggregate_config,
            db_engine=engine,
        ).generate()

        for output_table in output_tables:
            result = engine.execute(
                'select * from {} order by date, entity_id'.format(output_table)
            )
            records = [row for row in result]
            logging.warning(records)
            logging.warning(expected_output[output_table])
            assert records == expected_output[output_table]
