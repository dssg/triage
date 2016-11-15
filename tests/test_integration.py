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

from collate import collate

with open(path.join(path.dirname(__file__), "config/database.yml")) as f:
    config = yaml.load(f)
engine = sqlalchemy.create_engine('postgres://', connect_args=config)

def test_engine():
    assert len(engine.execute("SELECT * FROM food_inspections").fetchall()) == 966

def test_simple_explicit_agg():
    agg = collate.Aggregate(""" "Results" = 'Fail'""",["count"])
    st = collate.SpacetimeAggregation([agg],
        from_obj = ex.table('food_inspections'),
        group_intervals = {ex.column('License #') : ["1 year", "2 years", "all"],
                           ex.column('Zip') : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"Inspection Date"')
    for group_by, sels in st.get_selects().items():
        for sel in sels:
            engine.execute(sel) # Just test that we can execute the query

def test_simple_lazy_agg():
    agg = collate.Aggregate(""" "Results" = 'Fail'""",["count"])
    st = collate.SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        group_intervals = {'"License #"':["1 year", "2 years", "all"],
                           '"Zip"' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"Inspection Date"')
    for group, sels in st.get_selects().items():
        for sel in sels:
            engine.execute(sel) # Just test that we can execute the query

def test_simple_creates():
    agg = collate.Aggregate(""" "Results" = 'Fail'""",["count"])
    st = collate.SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        group_intervals = {'"License #"':["1 year", "2 years", "all"],
                           '"Zip"' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"Inspection Date"')

    creates, drops, indexes = st.get_creates(), st.get_drops(), st.get_indexes()
    conn = engine.connect()
    trans = conn.begin()
    for group in st.groups:
        conn.execute(drops[group])
        conn.execute(creates[group])
        conn.execute(indexes[group])
    trans.commit()

def test_simple_create():
    agg = collate.Aggregate(""" "Results" = 'Fail'""",["count"])
    st = collate.SpacetimeAggregation([agg],
        from_obj = 'food_inspections',
        group_intervals = {'"License #"':["1 year", "2 years", "all"],
                           '"Zip"' : ["1 year"]},
        dates = ['2016-08-31', '2015-08-31'],
        date_column = '"Inspection Date"')

    creates, drops, indexes = st.get_creates(), st.get_drops(), st.get_indexes()

    conn = engine.connect()
    trans = conn.begin()
    for group in st.groups:
        conn.execute(drops[group])
        conn.execute(creates[group])
        conn.execute(indexes[group])

    conn.execute(st.get_drop())
    conn.execute(st.get_create())
    trans.commit()
