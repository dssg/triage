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
engine = sqlalchemy.create_engine('postgres://', connect_args=config)

def test_engine():
    assert len(engine.execute("SELECT * FROM food_inspections").fetchall()) == 966

def test_st_explicit_execute():
    agg = Aggregate("results='Fail'",["count"])
    st = SpacetimeAggregation([agg],
        from_obj = ex.table('food_inspections'),
        groups = {'license':ex.column('license_no'), 
            'zip':ex.column('zip')},
        intervals = {'license' : ["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = 'inspection_date',
        prefix='food_inspections')

    st.execute(engine.connect())

def test_st_lazy_execute():
    agg = Aggregate("results='Fail'",["count"])
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = {'license_no':["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"inspection_date"')

    st.execute(engine.connect())

def test_st_execute_broadcast_intervals():
    agg = Aggregate("results='Fail'",["count"])
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = ["1 year", "2 years", "all"],
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"inspection_date"')

    st.execute(engine.connect())

def test_execute():
    agg = Aggregate("results='Fail'",["count"])
    st = Aggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'])

    st.execute(engine.connect())

def test_execute_schema_output_date_column():
    agg = Aggregate("results='Fail'",["count"])
    st = SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        groups = ['license_no', 'zip'],
        intervals = {'license_no':["1 year", "2 years", "all"],
                           'zip' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        schema = "agg",
        date_column = '"inspection_date"',
        output_date_column = "aggregation_date")

    st.execute(engine.connect())
