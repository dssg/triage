# -*- coding: utf-8 -*-
"""Integration tests for `collate` module."""

import pytest
from sqlalchemy import text
from sqlalchemy.sql import expression as ex

from triage.component.collate import Aggregate, Aggregation
from triage.component.collate.spacetime import SpacetimeAggregation

from . import initialize_db

IMPUTE_RULES = {
    "coltype": "aggregate",
    "count": {"type": "mean"},
    "mode": {"type": "mean"},
}


@pytest.fixture
def db_engine_with_collate_data(db_engine):
    """Create a db_engine with food inspections data loaded."""
    initialize_db.load_data(db_engine)
    return db_engine


def test_engine(db_engine_with_collate_data):
    with db_engine_with_collate_data.connect() as conn:
        ((result,),) = conn.execute(text("SELECT COUNT(*) FROM food_inspections"))
    assert result == 966


def test_st_explicit_execute(db_engine_with_collate_data):
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
    st.execute(db_engine_with_collate_data)


def test_st_lazy_execute(db_engine_with_collate_data):
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
    st.execute(db_engine_with_collate_data)


def test_st_execute_broadcast_intervals(db_engine_with_collate_data):
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
    st.execute(db_engine_with_collate_data)


def test_execute(db_engine_with_collate_data):
    agg = Aggregate("results='Fail'", ["count"], IMPUTE_RULES)
    st = Aggregation(
        [agg],
        from_obj="food_inspections",
        groups=["license_no", "zip"],
        state_table="all_licenses",
        state_group="license_no",
    )
    st.execute(db_engine_with_collate_data)


def test_execute_schema_output_date_column(db_engine_with_collate_data):
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
    st.execute(db_engine_with_collate_data)
