from architect.label_generators import BinaryLabelGenerator
from architect.state_table_generators import StateFilter
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date, timedelta

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


state_data = [
    # entity id, as_of_time, state_one, state_two
    [1, date(2014, 9, 30), True, False],
    [2, date(2014, 9, 30), True, True],
    [3, date(2014, 9, 30), False, True],
    [4, date(2014, 9, 30), True, False],
]


def create_events(engine):
    engine.execute(
        'create table events (entity_id int, outcome_date date, outcome bool)'
    )
    for event in events_data:
        engine.execute(
            'insert into events values (%s, %s, %s::bool)',
            event
        )


def create_sparse_state_table(engine):
    engine.execute('''create table sparse_states (
        entity_id int,
        as_of_time timestamp,
        state_one bool,
        state_two bool
    )''')

    for row in state_data:
        engine.execute(
            'insert into sparse_states values (%s, %s, %s, %s)',
            row
        )


def test_training_label_generation_no_state_table():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_events(engine)

        labels_table_name = 'labels'

        label_generator = BinaryLabelGenerator(
            events_table='events',
            db_engine=engine,
        )
        label_generator._create_labels_table(labels_table_name)
        label_generator.generate(
            start_date='2014-09-30',
            label_window='6month',
            labels_table=labels_table_name
        )

        result = engine.execute(
            'select * from {} order by entity_id, as_of_date'.format(labels_table_name)
        )
        records = [row for row in result]

        expected = [
            # entity_id, as_of_date, label_window, name, type, label
            (1, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
            (3, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', True),
            (4, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
        ]

        assert records == expected


def test_training_label_generation_with_state_table():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_events(engine)
        create_sparse_state_table(engine)

        labels_table_name = 'labels'

        label_generator = BinaryLabelGenerator(
            events_table='events',
            db_engine=engine,
        )

        state_filter = StateFilter(
            sparse_state_table='sparse_states',
            filter_logic='state_one and not state_two'
        )
        label_generator._create_labels_table(labels_table_name)
        label_generator.generate(
            start_date='2014-09-30',
            label_window='6month',
            labels_table=labels_table_name,
            state_filter=state_filter
        )

        result = engine.execute(
            'select * from {} order by entity_id, as_of_date'.format(labels_table_name)
        )
        records = [row for row in result]

        expected = [
            # entity_id, as_of_date, label_window, name, type, label
            (1, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
            (4, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
        ]

        assert records == expected


def test_generate_all_labels():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_events(engine)

        labels_table_name = 'labels'

        label_generator = BinaryLabelGenerator(
            events_table='events',
            db_engine=engine,
        )
        label_generator.generate_all_labels(
            labels_table=labels_table_name,
            as_of_times=['2014-09-30', '2015-03-30'],
            label_windows=['6month', '3month'],
        )

        result = engine.execute('''
            select * from {}
            order by entity_id, as_of_date, label_window desc
        '''.format(labels_table_name)
        )
        records = [row for row in result]

        expected = [
            # entity_id, as_of_date, label_window, name, type, label
            (1, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
            (1, date(2014, 9, 30), timedelta(90), 'outcome', 'binary', False),
            (2, date(2015, 3, 30), timedelta(180), 'outcome', 'binary', False),
            (2, date(2015, 3, 30), timedelta(90), 'outcome', 'binary', False),
            (3, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', True),
            (3, date(2015, 3, 30), timedelta(180), 'outcome', 'binary', False),
            (4, date(2014, 9, 30), timedelta(180), 'outcome', 'binary', False),
            (4, date(2014, 9, 30), timedelta(90), 'outcome', 'binary', False),
        ]
        assert records == expected
