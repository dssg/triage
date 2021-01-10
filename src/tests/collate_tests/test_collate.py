# -*- coding: utf-8 -*-
"""test_collate

Unit tests for `collate` module.

"""
import pytest
from triage.component.collate import Aggregate, Aggregation, Categorical

def test_aggregate():
    agg = Aggregate("*", "count", {})
    assert list(map(str, agg.get_columns())) == ["count(*)"]

def test_aggregate_cast():
    agg = Aggregate("*", "count", {}, coltype="REAL")
    assert list(map(str, agg.get_columns())) == ["count(*)::REAL"]

def test_categorical_cast():
    cat = Categorical("c", ['A','B','C'], "sum", {}, coltype="SMALLINT")
    assert list(map(str, cat.get_columns())) == [
        "sum((c = 'A')::INT)::SMALLINT",
        "sum((c = 'B')::INT)::SMALLINT",
        "sum((c = 'C')::INT)::SMALLINT"
    ]

def test_aggregate_when():
    agg = Aggregate("1", "count", {})
    assert list(map(str, agg.get_columns(when="date < '2012-01-01'"))) == [
        "count(1) FILTER (WHERE date < '2012-01-01')"
    ]

def test_aggregate_when_cast():
    agg = Aggregate("", "mode", {}, "x", coltype="SMALLINT")
    assert list(map(str, agg.get_columns(when="date < '2012-01-01'"))) == [
        "mode() WITHIN GROUP (ORDER BY x) FILTER (WHERE date < '2012-01-01')::SMALLINT"
    ]


def test_ordered_aggregate():
    agg = Aggregate("", "mode", {}, "x")
    (expression,) = agg.get_columns()
    assert str(expression) == "mode() WITHIN GROUP (ORDER BY x)"
    assert expression.name == "x_mode"


def test_ordered_aggregate_when():
    agg = Aggregate("", "mode", {}, "x")
    assert list(map(str, agg.get_columns(when="date < '2012-01-01'"))) == [
        "mode() WITHIN GROUP (ORDER BY x) FILTER (WHERE date < '2012-01-01')"
    ]




def test_aggregate_tuple_quantity():
    agg = Aggregate(("x", "y"), "corr", {})
    assert list(map(str, agg.get_columns())) == ["corr(x, y)"]


def test_aggregate_tuple_quantity_when():
    agg = Aggregate(("x", "y"), "corr", {})
    assert list(map(str, agg.get_columns(when="date < '2012-01-01'"))) == [
        "corr(x, y) FILTER (WHERE date < '2012-01-01')"
    ]


def test_aggregate_imputation_lookup():
    agg = Aggregate(
        "a",
        ["avg", "sum"],
        {
            "coltype": "aggregate",
            "avg": {"type": "mean"},
            "sum": {"type": "constant", "value": 3},
            "max": {"type": "zero"},
        },
    )
    assert agg.column_imputation_lookup()["a_avg"]["type"] == "mean"
    assert agg.column_imputation_lookup()["a_avg"]["coltype"] == "aggregate"
    assert agg.column_imputation_lookup()["a_sum"]["type"] == "constant"
    assert agg.column_imputation_lookup()["a_sum"]["value"] == 3
    assert agg.column_imputation_lookup()["a_sum"]["coltype"] == "aggregate"


def test_aggregate_imputation_lookup_all():
    agg = Aggregate(
        "a",
        ["avg", "sum"],
        {
            "coltype": "aggregate",
            "all": {"type": "zero"},
            "sum": {"type": "constant", "value": 3},
            "max": {"type": "mean"},
        },
    )
    assert agg.column_imputation_lookup()["a_avg"]["type"] == "zero"
    assert agg.column_imputation_lookup()["a_avg"]["coltype"] == "aggregate"
    assert agg.column_imputation_lookup()["a_sum"]["type"] == "constant"
    assert agg.column_imputation_lookup()["a_sum"]["value"] == 3
    assert agg.column_imputation_lookup()["a_sum"]["coltype"] == "aggregate"


def test_aggregate_arithmetic():
    n = Aggregate("x", "sum", {})
    d = Aggregate("1", "count", {})
    m = Aggregate("y", "avg", {})

    (e,) = (n / d + m).get_columns(prefix="prefix_")
    assert str(e) == "((sum(x)*1.0 / count(1)) + avg(y))"
    assert e.name == "prefix_x_sum/1_count+y_avg"


def test_aggregate_format_kwargs():
    agg = Aggregate("'{collate_date}' - date", "min", {})
    assert list(
        map(str, agg.get_columns(format_kwargs={"collate_date": "2012-01-01"}))
    ) == ["min('2012-01-01' - date)"]


def test_aggregation_table_name_no_schema():
    # no schema
    assert (
        Aggregation(
            [], from_obj="source", groups=[], state_table="tbl"
        ).get_table_name()
        == '"source_aggregation"'
    )
    assert (
        Aggregation([], from_obj="source", groups=[], state_table="tbl").get_table_name(
            imputed=True
        )
        == '"source_aggregation_imputed"'
    )

    # prefix
    assert (
        Aggregation(
            [], from_obj="source", prefix="mysource", groups=[], state_table="tbl"
        ).get_table_name()
        == '"mysource_aggregation"'
    )
    assert (
        Aggregation(
            [], from_obj="source", prefix="mysource", groups=[], state_table="tbl"
        ).get_table_name(imputed=True)
        == '"mysource_aggregation_imputed"'
    )

    # schema
    assert (
        Aggregation(
            [], from_obj="source", schema="schema", groups=[], state_table="tbl"
        ).get_table_name()
        == '"schema"."source_aggregation"'
    )
    assert (
        Aggregation(
            [], from_obj="source", schema="schema", groups=[], state_table="tbl"
        ).get_table_name(imputed=True)
        == '"schema"."source_aggregation_imputed"'
    )


def test_distinct():
    assert list(map(str, Aggregate("distinct x", "count", {}).get_columns())) == [
        "count(distinct x)"
    ]

    assert list(
        map(
            str,
            Aggregate("distinct x", "count", {}).get_columns(
                when="date < '2012-01-01'"
            ),
        )
    ) == ["count(distinct x) FILTER (WHERE date < '2012-01-01')"]

    assert list(
        map(
            str,
            Aggregate("distinct(x)", "count", {}).get_columns(
                when="date < '2012-01-01'"
            ),
        )
    ) == ["count(distinct (x)) FILTER (WHERE date < '2012-01-01')"]

    assert list(
        map(
            str,
            Aggregate("distinct(x,y)", "count", {}).get_columns(
                when="date < '2012-01-01'"
            ),
        )
    ) == ["count(distinct (x,y)) FILTER (WHERE date < '2012-01-01')"]


def test_Aggregation_colname_aggregate_lookup():
    n = Aggregate("x", "sum", {})
    d = Aggregate("1", "count", {})
    m = Aggregate("y", "avg", {})
    aggregation = Aggregation(
        [n, d, m],
        groups=['entity_id'],
        from_obj="source",
        prefix="mysource",
        state_table="tbl"
    )
    assert aggregation.colname_aggregate_lookup == {
        'mysource_entity_id_x_sum': 'sum',
        'mysource_entity_id_1_count': 'count',
        'mysource_entity_id_y_avg': 'avg'
    }

def test_Aggregation_colname_agg_function():
    n = Aggregate("x", "sum", {})
    d = Aggregate("1", "count", {})
    m = Aggregate("y", "stddev_samp", {})
    aggregation = Aggregation(
        [n, d, m],
        groups=['entity_id'],
        from_obj="source",
        prefix="mysource",
        state_table="tbl"
    )

    assert aggregation.colname_agg_function('mysource_entity_id_x_sum') == 'sum'
    assert aggregation.colname_agg_function('mysource_entity_id_y_stddev_samp') == 'stddev_samp'


def test_Aggregation_imputation_flag_base():
    n = Aggregate("x", ["sum", "count"], {})
    m = Aggregate("y", "stddev_samp", {})
    aggregation = Aggregation(
        [n, m],
        groups=['entity_id'],
        from_obj="source",
        prefix="mysource",
        state_table="tbl"
    )

    assert aggregation.imputation_flag_base('mysource_entity_id_x_sum') == 'mysource_entity_id_x'
    assert aggregation.imputation_flag_base('mysource_entity_id_x_count') == 'mysource_entity_id_x'
    assert aggregation.imputation_flag_base('mysource_entity_id_y_stddev_samp') == 'mysource_entity_id_y_stddev_samp'
    with pytest.raises(KeyError):
        aggregation.imputation_flag_base('mysource_entity_id_x_stddev_samp')
