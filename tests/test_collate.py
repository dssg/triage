#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_collate
----------------------------------

Unit tests for `collate` module.
"""

import pytest
from collate import collate

def test_aggregate():
    agg = collate.Aggregate("*", "count")
    assert str(list(agg.get_columns())[0]) == "count(*)"

def test_aggregate_when():
    agg = collate.Aggregate("1", "count")
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "count(CASE WHEN date < '2012-01-01' THEN 1 END)")

def test_ordered_aggregate():
    agg = collate.Aggregate("", "mode", "x")
    assert str(list(agg.get_columns())[0]) == "mode() WITHIN GROUP (ORDER BY x)"

def test_ordered_aggregate_when():
    agg = collate.Aggregate("", "mode", "x")
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "mode() WITHIN GROUP (ORDER BY CASE WHEN date < '2012-01-01' THEN x END)")

def test_aggregate_tuple_quantity():
    agg = collate.Aggregate(("x","y"), "corr")
    assert str(list(agg.get_columns())[0]) == "corr(x, y)"

def test_aggregate_tuple_quantity_when():
    agg = collate.Aggregate(("x","y"), "corr")
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "corr(CASE WHEN date < '2012-01-01' THEN x END, "
            "CASE WHEN date < '2012-01-01' THEN y END)")
