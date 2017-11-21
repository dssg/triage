#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_integration
----------------------------------

Integration tests for `collate` module.
"""

import yaml
import sqlalchemy
from os import path
import sqlalchemy.sql.expression as ex

from collate.collate import Aggregation, Aggregate
from collate.spacetime import SpacetimeAggregation

with open(path.join(path.dirname(__file__), "config/database.yml")) as f:
    config = yaml.load(f)

def test_engine():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    assert len(engine.execute("SELECT * FROM food_inspections").fetchall()) == 966

def test_st_explicit_execute():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    impute_rules={
        'coltype': 'aggregate', 
        'count': {'type': 'mean'},
        'mode': {'type': 'mean'}
        }
    agg = Aggregate({'F': "results='Fail'"},["count"],impute_rules)
    mode = Aggregate("", "mode", impute_rules, order="zip")
    st = SpacetimeAggregation([agg, agg+agg, mode],
        from_obj = ex.table('food_inspections'),
        groups = {'license':ex.column('license_no'), 
            'zip':ex.column('zip')},
        intervals = {'license' : ["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-30', '2015-11-06'],
        state_table = 'inspection_states',
        state_group = 'license_no',
        date_column = 'inspection_date',
        prefix='food_inspections')

    st.execute(engine.connect())

IMPUTE_RULES={
    'coltype': 'aggregate', 
    'count': {'type': 'mean'},
    'mode': {'type': 'mean'}
}

def test_st_lazy_execute():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    agg = Aggregate("results='Fail'",["count"],IMPUTE_RULES)
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = {'license_no':["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-30', '2015-11-06'],
        state_table = 'inspection_states',
        state_group = 'license_no',
        date_column = '"inspection_date"')

    st.execute(engine.connect())

def test_st_execute_broadcast_intervals():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    agg = Aggregate("results='Fail'",["count"], IMPUTE_RULES)
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = ["1 year", "2 years", "all"],
        dates = ['2016-08-30', '2015-11-06'],
        state_table = 'inspection_states',
        state_group = 'license_no',
        date_column = '"inspection_date"')

    st.execute(engine.connect())

def test_execute():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    agg = Aggregate("results='Fail'",["count"], IMPUTE_RULES)
    st = Aggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        state_table = 'all_licenses',
        state_group = 'license_no')

    st.execute(engine.connect())

def test_execute_schema_output_date_column():
    engine = sqlalchemy.create_engine('postgres://', connect_args=config)
    agg = Aggregate("results='Fail'",["count"], IMPUTE_RULES)
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = {'license_no':["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-30', '2015-11-06'],
        state_table = 'inspection_states_diff_colname',
        state_group = 'license_no',
        schema = "agg",
        date_column = '"inspection_date"',
        output_date_column = "aggregation_date")

    st.execute(engine.connect())
