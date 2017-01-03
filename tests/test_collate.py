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

def test_aggregate_format_kwargs():
    agg = collate.Aggregate("'{collate_date}' - date", "min")
    assert str(list(agg.get_columns(format_kwargs={"collate_date":"2012-01-01"}))[0]) == (
            "min('2012-01-01' - date)")

def test_aggregation_table_name_no_schema():
    # no schema
    assert collate.Aggregation([], from_obj='source', groups=[])\
            .get_table_name() == '"source_aggregation"'

    # prefix
    assert collate.Aggregation([], from_obj='source', prefix="mysource",
            groups=[])\
            .get_table_name() == '"mysource_aggregation"'

    # schema
    assert collate.Aggregation([], from_obj='source', schema='schema',
            groups=[])\
            .get_table_name() == '"schema"."source_aggregation"'

def test_distinct():
    assert str(list(collate.Aggregate("distinct x", "count").get_columns())[0]) == "count(distinct x)"

    assert str(list(collate.Aggregate("distinct x", "count").get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct CASE WHEN date < '2012-01-01' THEN x END)"

    assert str(list(collate.Aggregate("distinct(x)", "count").get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct CASE WHEN date < '2012-01-01' THEN (x) END)"

    assert str(list(collate.Aggregate("distinct(x,y)", "count").get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct CASE WHEN date < '2012-01-01' THEN (x,y) END)"
