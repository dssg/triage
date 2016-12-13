from triage.entity_feature_date_generators import \
    TimeOfEventFeatureDateGenerator, WholeWindowFeatureDateGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date

events_data = [
    # entity id, event_date, outcome
    [1, date(2014, 1, 1), 1],
    [1, date(2014, 11, 10), 0],
    [2, date(2014, 6, 8), 1],
    [3, date(2014, 3, 3), 0],
    [3, date(2014, 7, 24), 0],
    [4, date(2014, 12, 13), 0],
    [1, date(2015, 1, 1), 0],
    [1, date(2015, 11, 10), 1],
    [2, date(2015, 6, 8), 0],
    [3, date(2015, 3, 3), 1],
    [3, date(2015, 7, 24), 0],
    [4, date(2015, 12, 13), 0],
]

time_of_event_expected = [
    # entity_id, feature_date
    (1, date(2014, 11, 10)),
    (4, date(2014, 12, 13)),
    (1, date(2015, 1, 1)),
    (3, date(2015, 3, 3)),
]

whole_window_expected = [
    # entity_id, feature_date
    (1, date(2014, 9, 30)),
    (3, date(2014, 9, 30)),
    (4, date(2014, 9, 30)),
]

def test_time_of_event_feature_date_generation():
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
        output_table_name = TimeOfEventFeatureDateGenerator(
            split={
                'train_start': '2014-09-30',
                'train_end': '2015-03-30',
                'test_start': '2015-04-01',
                'test_end': '2015-10-01',
                'prediction_window': 6
            },
            events_table='events',
            db_engine=engine,
        ).generate()

        result = engine.execute(
            'select * from {} order by feature_date'.format(output_table_name)
        )
        records = [row for row in result]

        assert records == time_of_event_expected


def test_whole_window_feature_date_generation():
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
        output_table_name = WholeWindowFeatureDateGenerator(
            split={
                'train_start': '2014-09-30',
                'train_end': '2015-03-30',
                'test_start': '2015-04-01',
                'test_end': '2015-10-01',
                'prediction_window': 6
            },
            events_table='events',
            db_engine=engine,
        ).generate()

        result = engine.execute(
            'select * from {} order by entity_id'.format(output_table_name)
        )
        records = [row for row in result]
        assert records == whole_window_expected
