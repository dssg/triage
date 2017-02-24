from triage.label_generators import LabelGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
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
    # entity_id, outcome_date, outcome
    (1, date(2014, 9, 30), False),
    (2, date(2014, 9, 30), False),
    (3, date(2014, 9, 30), True),
    (4, date(2014, 9, 30), False),
]

def test_training_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        engine.execute(
            'create table events (entity_id int, date date, outcome bool)'
        )
        for event in events_data:
            engine.execute(
                'insert into events values (%s, %s, %s::bool)',
                event
            )

        labels_table_name = 'labels'
        engine.execute(
            'create table labels (entity_id int, outcome_date date, outcome bool)'
        )

        LabelGenerator(
            events_table='events',
            db_engine=engine,
        ).generate(
            start_date='2014-09-30',
            end_date='2015-09-30',
            labels_table=labels_table_name
        )

        result = engine.execute(
            'select * from {} order by entity_id, outcome_date'.format(labels_table_name)
        )
        records = [row for row in result]
        assert records == expected
