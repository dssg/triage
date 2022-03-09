from datetime import datetime, timedelta

import pytest
import testing.postgresql
from sqlalchemy.engine import create_engine

from triage.component.architect.entity_date_table_generators import EntityDateTableGenerator

from . import utils


def test_empty_output():
    """An empty cohort table eagerly produces an error.

    (Rather than allowing execution to proceed.)

    """
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", [])
        table_generator = EntityDateTableGenerator(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            entity_date_table_name="exp_hash_cohort",
        )

        with pytest.raises(ValueError):
            # Request time outside of available intervals
            table_generator.generate_entity_date_table([datetime(2015, 12, 31)])

        (cohort_count,) = engine.execute(
            f"""\
            select count(*) from {table_generator.entity_date_table_name}
        """
        ).first()

        assert cohort_count == 0

        engine.dispose()


def test_entity_date_table_generator_replace():
    input_data = [
        (1, datetime(2016, 1, 1), True),
        (1, datetime(2016, 4, 1), False),
        (1, datetime(2016, 3, 1), True),
        (2, datetime(2016, 1, 1), False),
        (2, datetime(2016, 1, 1), True),
        (3, datetime(2016, 1, 1), True),
        (5, datetime(2016, 3, 1), True),
        (5, datetime(2016, 4, 1), True),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", input_data)
        table_generator = EntityDateTableGenerator(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            entity_date_table_name="exp_hash_entity_date",
            replace=True
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_entity_date_table(as_of_dates)
        expected_output = [
            (1, datetime(2016, 2, 1), True),
            (1, datetime(2016, 3, 1), True),
            (1, datetime(2016, 4, 1), True),
            (1, datetime(2016, 5, 1), True),
            (1, datetime(2016, 6, 1), True),
            (2, datetime(2016, 2, 1), True),
            (2, datetime(2016, 3, 1), True),
            (2, datetime(2016, 4, 1), True),
            (2, datetime(2016, 5, 1), True),
            (2, datetime(2016, 6, 1), True),
            (3, datetime(2016, 2, 1), True),
            (3, datetime(2016, 3, 1), True),
            (3, datetime(2016, 4, 1), True),
            (3, datetime(2016, 5, 1), True),
            (3, datetime(2016, 6, 1), True),
            (5, datetime(2016, 4, 1), True),
            (5, datetime(2016, 5, 1), True),
            (5, datetime(2016, 6, 1), True),
        ]
        results = list(
            engine.execute(
                f"""
                select entity_id, as_of_date, active from {table_generator.entity_date_table_name}
                order by entity_id, as_of_date
            """
            )
        )
        assert results == expected_output
        utils.assert_index(engine, table_generator.entity_date_table_name, "entity_id")
        utils.assert_index(engine, table_generator.entity_date_table_name, "as_of_date")

        table_generator.generate_entity_date_table(as_of_dates)
        assert results == expected_output


def test_entity_date_table_generator_noreplace():
    input_data = [
        (1, datetime(2016, 1, 1), True),
        (1, datetime(2016, 4, 1), False),
        (1, datetime(2016, 3, 1), True),
        (2, datetime(2016, 1, 1), False),
        (2, datetime(2016, 1, 1), True),
        (3, datetime(2016, 1, 1), True),
        (5, datetime(2016, 3, 1), True),
        (5, datetime(2016, 4, 1), True),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", input_data)
        table_generator = EntityDateTableGenerator(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            entity_date_table_name="exp_hash_entity_date",
            replace=False
        )

        # 1. generate a cohort for a subset of as-of-dates
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
        ]
        table_generator.generate_entity_date_table(as_of_dates)
        expected_output = [
            (1, datetime(2016, 2, 1), True),
            (1, datetime(2016, 3, 1), True),
            (2, datetime(2016, 2, 1), True),
            (2, datetime(2016, 3, 1), True),
            (3, datetime(2016, 2, 1), True),
            (3, datetime(2016, 3, 1), True),
        ]
        results = list(
            engine.execute(
                f"""
                select entity_id, as_of_date, active from {table_generator.entity_date_table_name}
                order by entity_id, as_of_date
            """
            )
        )
        assert results == expected_output
        utils.assert_index(engine, table_generator.entity_date_table_name, "entity_id")
        utils.assert_index(engine, table_generator.entity_date_table_name, "as_of_date")

        table_generator.generate_entity_date_table(as_of_dates)
        assert results == expected_output

        # 2. generate a cohort for a different subset of as-of-dates,
        # actually including an overlap to make sure that it doesn't double-insert anything
        as_of_dates = [
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_entity_date_table(as_of_dates)
        expected_output = [
            (1, datetime(2016, 2, 1), True),
            (1, datetime(2016, 3, 1), True),
            (1, datetime(2016, 4, 1), True),
            (1, datetime(2016, 5, 1), True),
            (1, datetime(2016, 6, 1), True),
            (2, datetime(2016, 2, 1), True),
            (2, datetime(2016, 3, 1), True),
            (2, datetime(2016, 4, 1), True),
            (2, datetime(2016, 5, 1), True),
            (2, datetime(2016, 6, 1), True),
            (3, datetime(2016, 2, 1), True),
            (3, datetime(2016, 3, 1), True),
            (3, datetime(2016, 4, 1), True),
            (3, datetime(2016, 5, 1), True),
            (3, datetime(2016, 6, 1), True),
            (5, datetime(2016, 4, 1), True),
            (5, datetime(2016, 5, 1), True),
            (5, datetime(2016, 6, 1), True),
        ]
        results = list(
            engine.execute(
                f"""
                select entity_id, as_of_date, active from {table_generator.entity_date_table_name}
                order by entity_id, as_of_date
            """
            )
        )
        assert results == expected_output


def test_entity_date_table_generator_from_labels():
    labels_data = [
        (1, datetime(2016, 1, 1), timedelta(180), 'outcome', 'binary', 0),
        (1, datetime(2016, 4, 1), timedelta(180), 'outcome', 'binary', 1),
        (1, datetime(2016, 3, 1), timedelta(180), 'outcome', 'binary', 0),
        (2, datetime(2016, 1, 1), timedelta(180), 'outcome', 'binary', 0),
        (2, datetime(2016, 1, 1), timedelta(180), 'outcome', 'binary', 1),
        (3, datetime(2016, 1, 1), timedelta(180), 'outcome', 'binary', 0),
        (5, datetime(2016, 3, 1), timedelta(180), 'outcome', 'binary', 0),
        (5, datetime(2016, 4, 1), timedelta(180), 'outcome', 'binary', 1),
        (1, datetime(2016, 1, 1), timedelta(90), 'outcome', 'binary', 0),
        (1, datetime(2016, 4, 1), timedelta(90), 'outcome', 'binary', 0),
        (1, datetime(2016, 3, 1), timedelta(90), 'outcome', 'binary', 1),
        (2, datetime(2016, 1, 1), timedelta(90), 'outcome', 'binary', 0),
        (2, datetime(2016, 1, 1), timedelta(90), 'outcome', 'binary', 1),
        (3, datetime(2016, 1, 1), timedelta(90), 'outcome', 'binary', 0),
        (5, datetime(2016, 3, 1), timedelta(90), 'outcome', 'binary', 0),
        (5, datetime(2016, 4, 1), timedelta(90), 'outcome', 'binary', 0),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        labels_table_name = utils.create_labels(engine, labels_data)
        table_generator = EntityDateTableGenerator(
            query=None,
            labels_table_name=labels_table_name,
            db_engine=engine,
            entity_date_table_name="exp_hash_entity_date",
            replace=False
        )
        table_generator.generate_entity_date_table([])
        expected_output = [
            (1, datetime(2016, 1, 1)),
            (1, datetime(2016, 3, 1)),
            (1, datetime(2016, 4, 1)),
            (2, datetime(2016, 1, 1)),
            (3, datetime(2016, 1, 1)),
            (5, datetime(2016, 3, 1)),
            (5, datetime(2016, 4, 1)),
        ]
        results = list(
            engine.execute(
                f"""
                select entity_id, as_of_date from {table_generator.entity_date_table_name}
                order by entity_id, as_of_date
            """
            )
        )
        assert results == expected_output
