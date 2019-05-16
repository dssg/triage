from datetime import datetime

import pytest
import testing.postgresql
from sqlalchemy.engine import create_engine

from triage.component.architect.protected_groups_generators import ProtectedGroupsGenerator

from . import utils


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
