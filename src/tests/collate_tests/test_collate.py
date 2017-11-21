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
    agg = collate.Aggregate("*", "count", {})
    assert str(list(agg.get_columns())[0]) == "count(*)"

def test_aggregate_when():
    agg = collate.Aggregate("1", "count", {})
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "count(1) FILTER (WHERE date < '2012-01-01')")

def test_ordered_aggregate():
    agg = collate.Aggregate("", "mode", {}, "x")
    assert str(list(agg.get_columns())[0]) == "mode() WITHIN GROUP (ORDER BY x)"
    assert list(agg.get_columns())[0].name == "x_mode"

def test_ordered_aggregate_when():
    agg = collate.Aggregate("", "mode", {}, "x")
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "mode() WITHIN GROUP (ORDER BY x) FILTER (WHERE date < '2012-01-01')")

def test_aggregate_tuple_quantity():
    agg = collate.Aggregate(("x","y"), "corr", {})
    assert str(list(agg.get_columns())[0]) == "corr(x, y)"

def test_aggregate_tuple_quantity_when():
    agg = collate.Aggregate(("x","y"), "corr", {})
    assert str(list(agg.get_columns(when="date < '2012-01-01'"))[0]) == (
            "corr(x, y) FILTER (WHERE date < '2012-01-01')")

def test_aggregate_imputation_lookup():
    agg = collate.Aggregate("a", ["avg", "sum"], {
        "coltype": "aggregate",
        "avg": {"type": "mean"}, 
        "sum": {"type": "constant", "value": 3},
        "max": {"type": "zero"}
        })
    assert agg.column_imputation_lookup()['a_avg']['type'] == 'mean'
    assert agg.column_imputation_lookup()['a_avg']['coltype'] == 'aggregate'
    assert agg.column_imputation_lookup()['a_sum']['type'] == 'constant'
    assert agg.column_imputation_lookup()['a_sum']['value'] == 3
    assert agg.column_imputation_lookup()['a_sum']['coltype'] == 'aggregate'

def test_aggregate_imputation_lookup_all():
    agg = collate.Aggregate("a", ["avg", "sum"], {
        "coltype": "aggregate",
        "all": {"type": "zero"}, 
        "sum": {"type": "constant", "value": 3},
        "max": {"type": "mean"}
        })
    assert agg.column_imputation_lookup()['a_avg']['type'] == 'zero'
    assert agg.column_imputation_lookup()['a_avg']['coltype'] == 'aggregate'
    assert agg.column_imputation_lookup()['a_sum']['type'] == 'constant'
    assert agg.column_imputation_lookup()['a_sum']['value'] == 3
    assert agg.column_imputation_lookup()['a_sum']['coltype'] == 'aggregate'

def test_aggregate_arithmetic():
    n = collate.Aggregate("x", "sum", {})
    d = collate.Aggregate("1", "count", {})
    m = collate.Aggregate("y", "avg", {})

    e = list((n/d + m).get_columns(prefix="prefix_"))[0]
    assert str(e) == "((sum(x)*1.0 / count(1)) + avg(y))"
    assert e.name == "prefix_x_sum/1_count+y_avg"

def test_aggregate_format_kwargs():
    agg = collate.Aggregate("'{collate_date}' - date", "min", {})
    assert str(list(agg.get_columns(format_kwargs={"collate_date":"2012-01-01"}))[0]) == (
            "min('2012-01-01' - date)")

def test_aggregation_table_name_no_schema():
    # no schema
    assert collate.Aggregation([], from_obj='source', groups=[], state_table='tbl')\
            .get_table_name() == '"source_aggregation"'
    assert collate.Aggregation([], from_obj='source', groups=[], state_table='tbl')\
            .get_table_name(imputed=True) == '"source_aggregation_imputed"'

    # prefix
    assert collate.Aggregation([], from_obj='source', prefix="mysource",
            groups=[], state_table='tbl')\
            .get_table_name() == '"mysource_aggregation"'
    assert collate.Aggregation([], from_obj='source', prefix="mysource",
            groups=[], state_table='tbl')\
            .get_table_name(imputed=True) == '"mysource_aggregation_imputed"'

    # schema
    assert collate.Aggregation([], from_obj='source', schema='schema',
            groups=[], state_table='tbl')\
            .get_table_name() == '"schema"."source_aggregation"'
    assert collate.Aggregation([], from_obj='source', schema='schema',
            groups=[], state_table='tbl')\
            .get_table_name(imputed=True) == '"schema"."source_aggregation_imputed"'

def test_distinct():
    assert str(list(collate.Aggregate("distinct x", "count", {}).get_columns())[0]) == "count(distinct x)"

    assert str(list(collate.Aggregate("distinct x", "count", {}).get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct x) FILTER (WHERE date < '2012-01-01')"

    assert str(list(collate.Aggregate("distinct(x)", "count", {}).get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct (x)) FILTER (WHERE date < '2012-01-01')"

    assert str(list(collate.Aggregate("distinct(x,y)", "count", {}).get_columns(when="date < '2012-01-01'"))[0]) == "count(distinct (x,y)) FILTER (WHERE date < '2012-01-01')"
