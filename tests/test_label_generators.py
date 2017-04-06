from triage.label_generators import BinaryLabelGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date, timedelta
import logging

events_data = [
    # entity id, event_date, outcome
    [1, date(2014, 1, 1), True],
    [1, date(2014, 11, 10), False],
    [1, date(2015, 1, 1), False],
    [1, date(2015, 11, 10), True],
    [2, date(2014, 6, 8), True],
    [2, date(2015, 6, 8), False],
    [3, date(2014, 3, 3), False],
    [3, date(2014, 7, 24), False],
    [3, date(2015, 3, 3), True],
    [3, date(2015, 7, 24), False],
    [4, date(2014, 12, 13), False],
    [4, date(2015, 12, 13), False],
]

expected = [
    # entity_id, as_of_date, prediction_window, name, type, label
    (1, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
    (3, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', True),
    (4, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
]


def test_training_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        engine.execute(
            'create table events (entity_id int, outcome_date date, outcome bool)'
        )
        for event in events_data:
            engine.execute(
                'insert into events values (%s, %s, %s::bool)',
                event
            )

        labels_table_name = 'labels'

        label_generator = BinaryLabelGenerator(
            events_table='events',
            db_engine=engine,
        )
        label_generator._create_labels_table(labels_table_name)
        label_generator.generate(
            start_date='2014-09-30',
            prediction_window='6month',
            labels_table=labels_table_name
        )

        result = engine.execute(
            'select * from {} order by entity_id, as_of_date'.format(labels_table_name)
        )
        records = [row for row in result]
        assert records == expected
