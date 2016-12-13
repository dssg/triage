from triage.training_label_generators import TrainingLabelGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import logging

events_data = [
    # entity id, event_date, outcome
    [1, '2014-01-01', 1],
    [1, '2014-11-10', 0],
    [2, '2014-06-08', 1],
    [3, '2014-03-03', 0],
    [3, '2014-07-24', 0],
    [4, '2014-12-13', 0],
    [1, '2015-01-01', 0],
    [1, '2015-11-10', 1],
    [2, '2015-06-08', 0],
    [3, '2015-03-03', 1],
    [3, '2015-07-24', 0],
    [4, '2015-12-13', 0],
]

entity_feature_dates = [
    # entity_id, feature_date
    (1, date(2014, 11, 10)),
    (4, date(2014, 12, 13)),
    (1, date(2015, 1, 1)),
    (3, date(2015, 3, 3)),
]

expected = [
    # entity_id, event_date, outcome
    (1, date(2014, 11, 10), False),
    (1, date(2015, 1, 1), False),
    (3, date(2015, 3, 3), True),
    (4, date(2014, 12, 13), False),
]

def test_training_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        engine.execute(
            'create table events (entity_id int, event_date date, outcome bool)'
        )
        for event in events_data:
            engine.execute(
                'insert into events values (%s, %s, %s::bool)',
                event
            )

        engine.execute(
            'create table entity_feature_dates (entity_id int, feature_date date)'
        )
        for row in entity_feature_dates:
            engine.execute(
                'insert into entity_feature_dates values (%s, %s)',
                row
            )

        output_table_name = TrainingLabelGenerator(
            events_table='events',
            entity_feature_dates_table='entity_feature_dates',
            db_engine=engine,
        ).generate()

        result = engine.execute(
            'select * from {} order by entity_id, event_date'.format(output_table_name)
        )
        records = [row for row in result]
        assert len(records) == 4
        logging.warning(records)
        assert records == expected
