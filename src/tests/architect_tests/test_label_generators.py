from datetime import date, timedelta

import testing.postgresql
from sqlalchemy import create_engine

from triage.component.architect.label_generators import LabelGenerator

from .utils import create_binary_outcome_events

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


def test_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        labels_table_name = "labels"
        query = """select
            events.entity_id,
            bool_or(outcome::bool)::integer as outcome
        from events
        where
            '{as_of_date}' <= outcome_date
            and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
            group by entity_id
        """
        label_generator = LabelGenerator(db_engine=engine, query=query)
        label_generator._create_labels_table(labels_table_name)
        label_generator.generate(
            start_date="2014-09-30", label_timespan="6months", labels_table="labels"
        )

        expected = [
            # entity_id, as_of_date, label_timespan, name, type, label
            (1, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
            (3, date(2014, 9, 30), timedelta(180), "outcome", "binary", True),
            (4, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
        ]
        result = engine.execute(
            "select * from {} order by entity_id, as_of_date".format(labels_table_name)
        )
        records = [row for row in result]
        assert records == expected


def test_generate_all_labels():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        labels_table_name = "labels"

        query = """select
            events.entity_id,
            bool_or(outcome::bool)::integer as outcome
        from events
        where
            '{as_of_date}' <= outcome_date
            and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
            group by entity_id
        """

        label_generator = LabelGenerator(db_engine=engine, query=query)
        label_generator.generate_all_labels(
            labels_table=labels_table_name,
            as_of_dates=["2014-09-30", "2015-03-30"],
            label_timespans=["6month", "3month"],
        )

        result = engine.execute(
            """
            select * from {}
            order by entity_id, as_of_date, label_timespan desc
        """.format(
                labels_table_name
            )
        )
        records = [row for row in result]

        expected = [
            # entity_id, as_of_date, label_timespan, name, type, label
            (1, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
            (1, date(2014, 9, 30), timedelta(90), "outcome", "binary", False),
            (2, date(2015, 3, 30), timedelta(180), "outcome", "binary", False),
            (2, date(2015, 3, 30), timedelta(90), "outcome", "binary", False),
            (3, date(2014, 9, 30), timedelta(180), "outcome", "binary", True),
            (3, date(2015, 3, 30), timedelta(180), "outcome", "binary", False),
            (4, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
            (4, date(2014, 9, 30), timedelta(90), "outcome", "binary", False),
        ]
        assert records == expected
