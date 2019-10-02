# -*- coding: utf-8 -*-
"""test_spacetime

Unit tests for `collate.spacetime` module.

"""
from datetime import date
from itertools import product

import pytest
import sqlalchemy
import testing.postgresql

from triage.component.collate import Aggregate, SpacetimeAggregation


events_data = [
    # entity id, event_date, outcome
    [1, date(2014, 1, 1), True],
    [1, date(2014, 11, 10), False],
    [1, date(2015, 1, 1), False],
    [1, date(2015, 11, 10), True],
    [2, date(2013, 6, 8), True],
    [2, date(2014, 6, 8), False],
    [3, date(2014, 3, 3), False],
    [3, date(2014, 7, 24), False],
    [3, date(2015, 3, 3), True],
    [3, date(2015, 7, 24), False],
    [4, date(2015, 12, 13), False],
    [4, date(2016, 12, 13), True],
]

# distinct entity_id, event_date pairs
state_data = sorted(
    list(
        product(
            set([l[0] for l in events_data]),
            set([l[1] for l in events_data] + [date(2016, 1, 1)]),
        )
    )
)


def test_basic_spacetime():
    with testing.postgresql.Postgresql() as psql:
        engine = sqlalchemy.create_engine(psql.url())
        engine.execute(
            "create table events (entity_id int, event_date date, outcome bool)"
        )
        for event in events_data:
            engine.execute("insert into events values (%s, %s, %s::bool)", event)

        engine.execute("create table states (entity_id int, as_of_date date)")
        for state in state_data:
            engine.execute("insert into states values (%s, %s)", state)

        agg = Aggregate(
            "outcome::int",
            ["sum", "avg", "stddev"],
            {
                "coltype": "aggregate",
                "avg": {"type": "mean"},
                "sum": {"type": "constant", "value": 3},
                "stddev": {"type": "constant", "value": 2},
            },
        )
        st = SpacetimeAggregation(
            aggregates=[agg],
            from_obj="events",
            groups=["entity_id"],
            intervals=["1y", "2y", "all"],
            dates=["2016-01-01", "2015-01-01"],
            state_table="states",
            state_group="entity_id",
            date_column="event_date",
            output_date_column="as_of_date",
        )

        st.execute(engine.connect())

        r = engine.execute(
            "select * from events_entity_id order by entity_id, as_of_date"
        )
        rows = [x for x in r]
        assert rows[0]["entity_id"] == 1
        assert rows[0]["as_of_date"] == date(2015, 1, 1)
        assert rows[0]["events_entity_id_1y_outcome::int_sum"] == 1
        assert rows[0]["events_entity_id_1y_outcome::int_avg"] == 0.5
        assert rows[0]["events_entity_id_2y_outcome::int_sum"] == 1
        assert rows[0]["events_entity_id_2y_outcome::int_avg"] == 0.5
        assert rows[0]["events_entity_id_all_outcome::int_sum"] == 1
        assert rows[0]["events_entity_id_all_outcome::int_avg"] == 0.5
        assert rows[1]["entity_id"] == 1
        assert rows[1]["as_of_date"] == date(2016, 1, 1)
        assert rows[1]["events_entity_id_1y_outcome::int_sum"] == 1
        assert rows[1]["events_entity_id_1y_outcome::int_avg"] == 0.5
        assert rows[1]["events_entity_id_2y_outcome::int_sum"] == 2
        assert rows[1]["events_entity_id_2y_outcome::int_avg"] == 0.5
        assert rows[1]["events_entity_id_all_outcome::int_sum"] == 2
        assert rows[1]["events_entity_id_all_outcome::int_avg"] == 0.5

        assert rows[2]["entity_id"] == 2
        assert rows[2]["as_of_date"] == date(2015, 1, 1)
        assert rows[2]["events_entity_id_1y_outcome::int_sum"] == 0
        assert rows[2]["events_entity_id_1y_outcome::int_avg"] == 0
        assert rows[2]["events_entity_id_2y_outcome::int_sum"] == 1
        assert rows[2]["events_entity_id_2y_outcome::int_avg"] == 0.5
        assert rows[2]["events_entity_id_all_outcome::int_sum"] == 1
        assert rows[2]["events_entity_id_all_outcome::int_avg"] == 0.5
        assert rows[3]["entity_id"] == 2
        assert rows[3]["as_of_date"] == date(2016, 1, 1)
        assert rows[3]["events_entity_id_1y_outcome::int_sum"] is None
        assert rows[3]["events_entity_id_1y_outcome::int_avg"] is None
        assert rows[3]["events_entity_id_2y_outcome::int_sum"] == 0
        assert rows[3]["events_entity_id_2y_outcome::int_avg"] == 0
        assert rows[3]["events_entity_id_all_outcome::int_sum"] == 1
        assert rows[3]["events_entity_id_all_outcome::int_avg"] == 0.5

        assert rows[4]["entity_id"] == 3
        assert rows[4]["as_of_date"] == date(2015, 1, 1)
        assert rows[4]["events_entity_id_1y_outcome::int_sum"] == 0
        assert rows[4]["events_entity_id_1y_outcome::int_avg"] == 0
        assert rows[4]["events_entity_id_2y_outcome::int_sum"] == 0
        assert rows[4]["events_entity_id_2y_outcome::int_avg"] == 0
        assert rows[4]["events_entity_id_all_outcome::int_sum"] == 0
        assert rows[4]["events_entity_id_all_outcome::int_avg"] == 0
        assert rows[5]["entity_id"] == 3
        assert rows[5]["as_of_date"] == date(2016, 1, 1)
        assert rows[5]["events_entity_id_1y_outcome::int_sum"] == 1
        assert rows[5]["events_entity_id_1y_outcome::int_avg"] == 0.5
        assert rows[5]["events_entity_id_2y_outcome::int_sum"] == 1
        assert rows[5]["events_entity_id_2y_outcome::int_avg"] == 0.25
        assert rows[5]["events_entity_id_all_outcome::int_sum"] == 1
        assert rows[5]["events_entity_id_all_outcome::int_avg"] == 0.25

        assert rows[6]["entity_id"] == 4
        # rows[6]['date'] == date(2015, 1, 1) is skipped due to no data!
        assert rows[6]["as_of_date"] == date(2016, 1, 1)
        assert rows[6]["events_entity_id_1y_outcome::int_sum"] == 0
        assert rows[6]["events_entity_id_1y_outcome::int_avg"] == 0
        assert rows[6]["events_entity_id_2y_outcome::int_sum"] == 0
        assert rows[6]["events_entity_id_2y_outcome::int_avg"] == 0
        assert rows[6]["events_entity_id_all_outcome::int_sum"] == 0
        assert rows[6]["events_entity_id_all_outcome::int_avg"] == 0
        assert len(rows) == 7

        # check some imputation results
        r = engine.execute(
            "select * from events_aggregation_imputed order by entity_id, as_of_date"
        )
        rows = [x for x in r]
        assert rows[6]["entity_id"] == 4
        assert rows[6]["as_of_date"] == date(2015, 1, 1)
        assert rows[6]["events_entity_id_1y_outcome::int_sum"] == 3
        assert rows[6]["events_entity_id_1y_outcome::int_imp"] == 1
        assert rows[6]["events_entity_id_1y_outcome::int_stddev"] == 2
        assert rows[6]["events_entity_id_1y_outcome::int_stddev_imp"] == 1
        assert (
            round(float(rows[6]["events_entity_id_1y_outcome::int_avg"]), 4) == 0.1667
        )
        assert rows[6]["events_entity_id_2y_outcome::int_sum"] == 3
        assert rows[6]["events_entity_id_2y_outcome::int_imp"] == 1
        assert rows[6]["events_entity_id_2y_outcome::int_stddev"] == 2
        assert rows[6]["events_entity_id_2y_outcome::int_stddev_imp"] == 1
        assert (
            round(float(rows[6]["events_entity_id_2y_outcome::int_avg"]), 4) == 0.3333
        )
        assert rows[6]["events_entity_id_all_outcome::int_sum"] == 3
        assert rows[6]["events_entity_id_all_outcome::int_imp"] == 1
        assert rows[6]["events_entity_id_all_outcome::int_stddev"] == 2
        assert rows[6]["events_entity_id_all_outcome::int_stddev_imp"] == 1
        assert (
            round(float(rows[6]["events_entity_id_all_outcome::int_avg"]), 4) == 0.3333
        )
        assert rows[6]["events_entity_id_all_outcome::int_imp"] == 1
        assert rows[7]["entity_id"] == 4
        assert rows[7]["as_of_date"] == date(2016, 1, 1)
        assert rows[7]["events_entity_id_1y_outcome::int_sum"] == 0
        assert rows[7]["events_entity_id_1y_outcome::int_imp"] == 0
        assert rows[7]["events_entity_id_1y_outcome::int_avg"] == 0
        assert rows[7]["events_entity_id_1y_outcome::int_stddev"] == 2
        assert rows[7]["events_entity_id_1y_outcome::int_stddev_imp"] == 1
        assert rows[7]["events_entity_id_2y_outcome::int_sum"] == 0
        assert rows[7]["events_entity_id_2y_outcome::int_imp"] == 0
        assert rows[7]["events_entity_id_2y_outcome::int_avg"] == 0
        assert rows[7]["events_entity_id_2y_outcome::int_stddev"] == 2
        assert rows[7]["events_entity_id_2y_outcome::int_stddev_imp"] == 1
        assert rows[7]["events_entity_id_all_outcome::int_sum"] == 0
        assert rows[7]["events_entity_id_all_outcome::int_imp"] == 0
        assert rows[7]["events_entity_id_all_outcome::int_avg"] == 0
        assert rows[7]["events_entity_id_all_outcome::int_stddev"] == 2
        assert rows[7]["events_entity_id_all_outcome::int_stddev_imp"] == 1
        assert len(rows) == 8


def test_input_min_date():
    with testing.postgresql.Postgresql() as psql:
        engine = sqlalchemy.create_engine(psql.url())
        engine.execute("create table events (entity_id int, date date, outcome bool)")
        for event in events_data:
            engine.execute("insert into events values (%s, %s, %s::bool)", event)

        engine.execute("create table states (entity_id int, date date)")
        for state in state_data:
            engine.execute("insert into states values (%s, %s)", state)

        agg = Aggregate(
            "outcome::int",
            ["sum", "avg"],
            {
                "coltype": "aggregate",
                "avg": {"type": "mean"},
                "sum": {"type": "constant", "value": 3},
                "max": {"type": "zero"},
            },
        )
        st = SpacetimeAggregation(
            aggregates=[agg],
            from_obj="events",
            groups=["entity_id"],
            intervals=["all"],
            dates=["2016-01-01"],
            state_table="states",
            state_group="entity_id",
            date_column='"date"',
            input_min_date="2015-11-10",
        )

        st.execute(engine.connect())

        r = engine.execute("select * from events_entity_id order by entity_id")
        rows = [x for x in r]

        assert rows[0]["entity_id"] == 1
        assert rows[0]["date"] == date(2016, 1, 1)
        assert rows[0]["events_entity_id_all_outcome::int_sum"] == 1
        assert rows[0]["events_entity_id_all_outcome::int_avg"] == 1
        assert rows[1]["entity_id"] == 4
        assert rows[1]["date"] == date(2016, 1, 1)
        assert rows[1]["events_entity_id_all_outcome::int_sum"] == 0
        assert rows[1]["events_entity_id_all_outcome::int_avg"] == 0

        assert len(rows) == 2

        st = SpacetimeAggregation(
            aggregates=[agg],
            from_obj="events",
            groups=["entity_id"],
            intervals=["1y", "all"],
            dates=["2016-01-01", "2015-01-01"],
            state_table="states",
            state_group="entity_id",
            date_column='"date"',
            input_min_date="2014-11-10",
        )
        with pytest.raises(ValueError):
            st.validate(engine.connect())
        with pytest.raises(ValueError):
            st.execute(engine.connect())


def test_join_with_cohort_table(db_engine):
    # if we specify joining with the cohort table
    # only entity_id/date pairs in the cohort table should show up
    db_engine.execute("create table events (entity_id int, date date, outcome bool)")
    for event in events_data:
        db_engine.execute("insert into events values (%s, %s, %s::bool)", event)

    db_engine.execute("create table cohort (entity_id int, date date)")

    # use the states list from above except only include entities 1 and 2 in the cohort
    smaller_cohort = sorted(
        product(
            set([l[0] for l in events_data if l[0] == 1 or l[0] == 2]),
            set([l[1] for l in events_data] + [date(2016, 1, 1)]),
        )
    )
    for state in smaller_cohort:
        db_engine.execute("insert into cohort values (%s, %s)", state)

    # create our test aggregation with the important 'join_with_cohort_table' flag
    agg = Aggregate(
        "outcome::int",
        ["sum", "avg"],
        {
            "coltype": "aggregate",
            "avg": {"type": "mean"},
            "sum": {"type": "constant", "value": 3},
            "max": {"type": "zero"},
        },
    )
    st = SpacetimeAggregation(
        aggregates=[agg],
        from_obj="events",
        groups=["entity_id"],
        intervals=["all"],
        dates=["2016-01-01", "2015-01-01"],
        state_table="cohort",
        state_group="entity_id",
        date_column='"date"',
        join_with_cohort_table=True,
    )

    st.execute(db_engine.connect())

    r = db_engine.execute("select * from events_entity_id order by entity_id, date")
    rows = [x for x in r]

    # these rows should be similar to the rows in the basic spacetime test,
    # except only the rows for entities 1 and 2 are present
    assert len(rows) == 4

    assert rows[0]["entity_id"] == 1
    assert rows[0]["date"] == date(2015, 1, 1)
    assert rows[0]["events_entity_id_all_outcome::int_sum"] == 1
    assert rows[0]["events_entity_id_all_outcome::int_avg"] == 0.5
    assert rows[1]["entity_id"] == 1
    assert rows[1]["date"] == date(2016, 1, 1)
    assert rows[1]["events_entity_id_all_outcome::int_sum"] == 2
    assert rows[1]["events_entity_id_all_outcome::int_avg"] == 0.5

    assert rows[2]["entity_id"] == 2
    assert rows[2]["date"] == date(2015, 1, 1)
    assert rows[2]["events_entity_id_all_outcome::int_sum"] == 1
    assert rows[2]["events_entity_id_all_outcome::int_avg"] == 0.5
    assert rows[3]["entity_id"] == 2
    assert rows[3]["date"] == date(2016, 1, 1)
    assert rows[3]["events_entity_id_all_outcome::int_sum"] == 1
    assert rows[3]["events_entity_id_all_outcome::int_avg"] == 0.5
