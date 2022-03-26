from datetime import date, timedelta

import testing.postgresql
import pytest
from sqlalchemy import create_engine

from triage.component.architect.label_generators import LabelGenerator

from .utils import create_binary_outcome_events


# Sample events data to use for all tests
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


# An example label generation query to use for all tests.
# Since this is expected to be passed in properly by the user
# we don't need to test variations. Just reuse this one.
LABEL_GENERATE_QUERY = """select
    events.entity_id,
    bool_or(outcome::bool)::integer as outcome
from events
where
    '{as_of_date}' <= outcome_date
    and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
    group by entity_id
"""

LABELS_TABLE_NAME = "labels"


def test_label_generation():
    # Generate labels for one as-of-date/label timespan combo
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        label_generator = LabelGenerator(db_engine=engine, query=LABEL_GENERATE_QUERY)
        label_generator._create_labels_table(LABELS_TABLE_NAME)
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
            "select * from {} order by entity_id, as_of_date".format(LABELS_TABLE_NAME)
        )
        records = [row for row in result]
        assert records == expected


def test_generate_all_labels_replace():
    # Generate labels for combinations of as-of-date and label timespan
    # use replace=True
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        label_generator = LabelGenerator(db_engine=engine, query=LABEL_GENERATE_QUERY, replace=True)
        label_generator.generate_all_labels(
            labels_table=LABELS_TABLE_NAME,
            as_of_dates=["2014-09-30", "2015-03-30"],
            label_timespans=["6month", "3month"],
        )

        result = engine.execute(
            """
            select * from {}
            order by entity_id, as_of_date, label_timespan desc
        """.format(
                LABELS_TABLE_NAME
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


def test_generate_all_labels_noreplace():
    # test the 'replace=False' functionality
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        label_generator = LabelGenerator(
            db_engine=engine,
            query=LABEL_GENERATE_QUERY,
            replace=False
        )
        label_generator.generate_all_labels(
            labels_table=LABELS_TABLE_NAME,
            as_of_dates=["2014-09-30"],
            label_timespans=["6month"],
        )

        result = engine.execute(
            """
            select * from {}
            order by entity_id, as_of_date, label_timespan desc
        """.format(
                LABELS_TABLE_NAME
            )
        )
        records = [row for row in result]

        expected = [
            # entity_id, as_of_date, label_timespan, name, type, label
            (1, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
            (3, date(2014, 9, 30), timedelta(180), "outcome", "binary", True),
            (4, date(2014, 9, 30), timedelta(180), "outcome", "binary", False),
        ]
        assert records == expected

        # round 2
        label_generator.generate_all_labels(
            labels_table=LABELS_TABLE_NAME,
            as_of_dates=["2014-09-30", "2015-03-30"],
            label_timespans=["6month", "3month"],
        )

        result = engine.execute(
            """
            select * from {}
            order by entity_id, as_of_date, label_timespan desc
        """.format(
                LABELS_TABLE_NAME
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


def test_generate_all_labels_errors_on_duplicates():

    # label query that will yield duplicates (one row for each event in the timespan)
    BAD_LABEL_GENERATE_QUERY = """
    select
        events.entity_id,
        1 as outcome
    from events
    where
        '{as_of_date}' <= outcome_date
        and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
    """

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, "events", events_data)

        label_generator = LabelGenerator(db_engine=engine, query=BAD_LABEL_GENERATE_QUERY, replace=True)
        with pytest.raises(ValueError):
            label_generator.generate_all_labels(
                labels_table=LABELS_TABLE_NAME,
                as_of_dates=["2014-09-30", "2015-03-30"],
                label_timespans=["6month", "3month"],
            )

