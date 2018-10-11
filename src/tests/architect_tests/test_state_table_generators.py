from datetime import datetime

import pytest
import testing.postgresql
from sqlalchemy.engine import create_engine

from triage.component.architect.state_table_generators import (
    StateTableGeneratorFromDense,
    StateTableGeneratorFromEntities,
    StateTableGeneratorFromQuery,
)

from . import utils


def test_sparse_state_table_generator():
    input_data = [
        (5, "permitted", datetime(2016, 1, 1), datetime(2016, 6, 1)),
        (6, "permitted", datetime(2016, 2, 5), datetime(2016, 5, 5)),
        (1, "injail", datetime(2014, 7, 7), datetime(2014, 7, 15)),
        (1, "injail", datetime(2016, 3, 7), datetime(2016, 4, 2)),
    ]

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_dense_state_table(engine, "states", input_data)

        table_generator = StateTableGeneratorFromDense(
            db_engine=engine, experiment_hash="exp_hash", dense_state_table="states"
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_sparse_table(as_of_dates)
        results = [
            row
            for row in engine.execute(
                "select entity_id, as_of_date, injail, permitted from {} "
                "order by entity_id, as_of_date".format(
                    table_generator.sparse_table_name
                )
            )
        ]
        expected_output = [
            # entity_id, as_of_date, injail, permitted
            (1, datetime(2016, 4, 1), True, False),
            (5, datetime(2016, 1, 1), False, True),
            (5, datetime(2016, 2, 1), False, True),
            (5, datetime(2016, 3, 1), False, True),
            (5, datetime(2016, 4, 1), False, True),
            (5, datetime(2016, 5, 1), False, True),
            (6, datetime(2016, 3, 1), False, True),
            (6, datetime(2016, 4, 1), False, True),
            (6, datetime(2016, 5, 1), False, True),
        ]
        assert results == expected_output
        utils.assert_index(engine, table_generator.sparse_table_name, "entity_id")
        utils.assert_index(engine, table_generator.sparse_table_name, "as_of_date")


def test_empty_dense_state_table():
    """An empty dense (input) state table produces a useful error."""
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_dense_state_table(engine, "states", ())  # no data
        table_generator = StateTableGeneratorFromDense(
            db_engine=engine, experiment_hash="exp_hash", dense_state_table="states"
        )

        with pytest.raises(ValueError):
            table_generator.generate_sparse_table([datetime(2016, 1, 1)])

        engine.dispose()


def test_empty_sparse_state_table():
    """An empty sparse (generated) state table eagerly produces an
    error.

    (Rather than allowing execution to proceed.)

    """
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_dense_state_table(
            engine,
            "states",
            (
                (5, "permitted", datetime(2016, 1, 1), datetime(2016, 6, 1)),
                (6, "permitted", datetime(2016, 2, 5), datetime(2016, 5, 5)),
                (1, "injail", datetime(2014, 7, 7), datetime(2014, 7, 15)),
                (1, "injail", datetime(2016, 3, 7), datetime(2016, 4, 2)),
            ),
        )
        table_generator = StateTableGeneratorFromDense(
            db_engine=engine, experiment_hash="exp_hash", dense_state_table="states"
        )

        with pytest.raises(ValueError):
            # Request time outside of available intervals
            table_generator.generate_sparse_table([datetime(2015, 12, 31)])

        (state_count,) = engine.execute(
            """\
            select count(*) from {generator.sparse_table_name}
        """.format(
                generator=table_generator
            )
        ).first()

        assert state_count == 0

        engine.dispose()


def test_sparse_table_generator_from_entities():
    input_data = [
        (1, datetime(2016, 1, 1), True),
        (1, datetime(2016, 4, 1), False),
        (1, datetime(2016, 3, 1), True),
        (2, datetime(2016, 1, 1), False),
        (2, datetime(2016, 1, 1), True),
        (3, datetime(2016, 1, 1), True),
        (5, datetime(2016, 1, 1), True),
        (5, datetime(2016, 1, 1), True),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", input_data)
        table_generator = StateTableGeneratorFromEntities(
            entities_table="events", db_engine=engine, experiment_hash="exp_hash"
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_sparse_table(as_of_dates)
        expected_output = [
            (1, datetime(2016, 1, 1), True),
            (1, datetime(2016, 2, 1), True),
            (1, datetime(2016, 3, 1), True),
            (1, datetime(2016, 4, 1), True),
            (1, datetime(2016, 5, 1), True),
            (1, datetime(2016, 6, 1), True),
            (2, datetime(2016, 1, 1), True),
            (2, datetime(2016, 2, 1), True),
            (2, datetime(2016, 3, 1), True),
            (2, datetime(2016, 4, 1), True),
            (2, datetime(2016, 5, 1), True),
            (2, datetime(2016, 6, 1), True),
            (3, datetime(2016, 1, 1), True),
            (3, datetime(2016, 2, 1), True),
            (3, datetime(2016, 3, 1), True),
            (3, datetime(2016, 4, 1), True),
            (3, datetime(2016, 5, 1), True),
            (3, datetime(2016, 6, 1), True),
            (5, datetime(2016, 1, 1), True),
            (5, datetime(2016, 2, 1), True),
            (5, datetime(2016, 3, 1), True),
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
                    table_generator.sparse_table_name
                )
            )
        ]
        assert results == expected_output
        utils.assert_index(engine, table_generator.sparse_table_name, "entity_id")
        utils.assert_index(engine, table_generator.sparse_table_name, "as_of_date")


def test_sparse_states_from_query():
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
        table_generator = StateTableGeneratorFromQuery(
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            db_engine=engine,
            experiment_hash="exp_hash",
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_sparse_table(as_of_dates)
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
                    table_generator.sparse_table_name
                )
            )
        ]
        assert results == expected_output
        utils.assert_index(engine, table_generator.sparse_table_name, "entity_id")
        utils.assert_index(engine, table_generator.sparse_table_name, "as_of_date")
