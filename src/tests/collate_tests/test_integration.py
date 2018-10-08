# -*- coding: utf-8 -*-
"""Integration tests for `collate` module."""
import testing.postgresql
from sqlalchemy import create_engine
from sqlalchemy.sql import expression as ex

from triage.component.collate import Aggregation, Aggregate
from triage.component.collate.spacetime import SpacetimeAggregation

from . import initialize_db


IMPUTE_RULES = {
    "coltype": "aggregate",
    "count": {"type": "mean"},
    "mode": {"type": "mean"},
}

Postgresql = testing.postgresql.PostgresqlFactory(
    cache_initialized_db=True, on_initialized=initialize_db.handler
)


def teardown_module():
    Postgresql.clear_cache()


def test_engine():
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ((result,),) = engine.execute("SELECT COUNT(*) FROM food_inspections")
    assert result == 966


def test_st_explicit_execute():
    agg = Aggregate({"F": "results='Fail'"}, ["count"], IMPUTE_RULES)
    mode = Aggregate("", "mode", IMPUTE_RULES, order="zip")
    st = SpacetimeAggregation(
        [agg, agg + agg, mode],
        from_obj=ex.table("food_inspections"),
        groups={"license": ex.column("license_no"), "zip": ex.column("zip")},
        intervals={"license": ["1 year", "2 years", "all"], "zip": ["1 year"]},
        dates=["2016-08-30", "2015-11-06"],
        state_table="inspection_states",
        state_group="license_no",
        date_column="inspection_date",
        prefix="food_inspections",
    )
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        st.execute(engine.connect())


def test_st_lazy_execute():
    agg = Aggregate("results='Fail'", ["count"], IMPUTE_RULES)
    st = SpacetimeAggregation(
        [agg],
        from_obj="food_inspections",
        groups=["license_no", "zip"],
        intervals={"license_no": ["1 year", "2 years", "all"], "zip": ["1 year"]},
        dates=["2016-08-30", "2015-11-06"],
        state_table="inspection_states",
        state_group="license_no",
        date_column='"inspection_date"',
    )
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        st.execute(engine.connect())


def test_st_execute_broadcast_intervals():
    agg = Aggregate("results='Fail'", ["count"], IMPUTE_RULES)
    st = SpacetimeAggregation(
        [agg],
        from_obj="food_inspections",
        groups=["license_no", "zip"],
        intervals=["1 year", "2 years", "all"],
        dates=["2016-08-30", "2015-11-06"],
        state_table="inspection_states",
        state_group="license_no",
        date_column='"inspection_date"',
    )
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        st.execute(engine.connect())


def test_execute():
    agg = Aggregate("results='Fail'", ["count"], IMPUTE_RULES)
    st = Aggregation(
        [agg],
        from_obj="food_inspections",
        groups=["license_no", "zip"],
        state_table="all_licenses",
        state_group="license_no",
    )
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        st.execute(engine.connect())


def test_execute_schema_output_date_column():
    agg = Aggregate("results='Fail'", ["count"], IMPUTE_RULES)
    st = SpacetimeAggregation(
        [agg],
        from_obj="food_inspections",
        groups=["license_no", "zip"],
        intervals={"license_no": ["1 year", "2 years", "all"], "zip": ["1 year"]},
        dates=["2016-08-30", "2015-11-06"],
        state_table="inspection_states_diff_colname",
        state_group="license_no",
        schema="agg",
        date_column='"inspection_date"',
        output_date_column="aggregation_date",
    )
    with Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        st.execute(engine.connect())
