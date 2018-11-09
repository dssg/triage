from datetime import datetime

import pytest
import testing.postgresql
from sqlalchemy.engine import create_engine

from triage.component.architect.cohort_table_generators import CohortTableGenerator

from . import utils


def test_empty_output():
    """An empty cohort table eagerly produces an error.

    (Rather than allowing execution to proceed.)

    """
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", [])
        table_generator = CohortTableGenerator(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            cohort_table_name="exp_hash_cohort",
        )

        with pytest.raises(ValueError):
            # Request time outside of available intervals
            table_generator.generate_cohort_table([datetime(2015, 12, 31)])

        (cohort_count,) = engine.execute(
            """\
            select count(*) from {generator.cohort_table_name}
        """.format(
                generator=table_generator
            )
        ).first()

        assert cohort_count == 0

        engine.dispose()


def test_cohort_table_generator():
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
        table_generator = CohortTableGenerator(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            cohort_table_name="exp_hash_cohort",
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_cohort_table(as_of_dates)
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
        results = [
            row
            for row in engine.execute(
                """
                select entity_id, as_of_date, active from {}
                order by entity_id, as_of_date
            """.format(
                    table_generator.cohort_table_name
                )
            )
        ]
        assert results == expected_output
        utils.assert_index(engine, table_generator.cohort_table_name, "entity_id")
        utils.assert_index(engine, table_generator.cohort_table_name, "as_of_date")
