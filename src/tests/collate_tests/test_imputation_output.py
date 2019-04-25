# -*- coding: utf-8 -*-
"""test_imputation_output

Unit tests for imputation output.

For all available imputation methods, make sure they correctly handle a variety
of cases, including completely null columns and columns entirely null for a given
date, generating the right number of records with no nulls in the output set and
imputed flags as necessary.

"""
import pandas as pd
import pytest
import sqlalchemy
import testing.postgresql

from triage.component.collate import (
    Aggregate,
    available_imputations,
    SpacetimeAggregation,
)

# some imputations require arguments, so specify default values here
# everything in collate.collate.available_imputations needs a record here!
imputation_values = {
    "mean": {
        "aggregate": {"avail": True, "kwargs": {}},
        "categorical": {"avail": True, "kwargs": {}},
    },
    "constant": {
        "aggregate": {"avail": True, "kwargs": {"value": 7}},
        "categorical": {"avail": True, "kwargs": {"value": "foo"}},
    },
    "zero": {
        "aggregate": {"avail": True, "kwargs": {}},
        "categorical": {"avail": True, "kwargs": {}},
    },
    "zero_noflag": {
        "aggregate": {"avail": True, "kwargs": {}},
        "categorical": {"avail": True, "kwargs": {}},
    },
    "null_category": {
        "aggregate": {"avail": False},
        "categorical": {"avail": True, "kwargs": {}},
    },
    "binary_mode": {
        "aggregate": {"avail": True, "kwargs": {}},
        "categorical": {"avail": False},
    },
}

states_table = [
    # entity_id, as_of_date
    [1, "2016-01-01"],
    [2, "2016-01-01"],
    [3, "2016-01-01"],
    [4, "2016-01-01"],
    [1, "2016-02-03"],
    [2, "2016-02-03"],
    [3, "2016-02-03"],
    [4, "2016-02-03"],
    [1, "2016-03-14"],
    [2, "2016-03-14"],
    [3, "2016-03-14"],
    [4, "2016-03-14"],
]

aggs_table = [
    # entity_id, as_of_date, f1, f2, f3, f4
    [1, "2016-01-01", None, 1, 3, 0],
    [2, "2016-01-01", None, 5, None, 0],
    [3, "2016-01-01", None, 7, 3, 0],
    [4, "2016-01-01", None, 4, 3, 0],
    [1, "2016-02-03", None, None, 3, 0],
    [2, "2016-02-03", None, None, 5, 8],
    [3, "2016-02-03", None, None, None, 2],
    [4, "2016-02-03", None, None, 4, 6],
    [1, "2016-03-14", None, 1, 3, 2],
    [4, "2016-03-14", None, 3, 9, 1],
]

aggs_table_noimp = [
    # entity_id, as_of_date, f5, f6
    [1, "2016-01-01", 3, 0],
    [2, "2016-01-01", None, 0],
    [3, "2016-01-01", 3, 0],
    [4, "2016-01-01", 3, 0],
    [1, "2016-02-03", 3, 0],
    [2, "2016-02-03", 5, 8],
    [3, "2016-02-03", None, 2],
    [4, "2016-02-03", 4, 6],
    [1, "2016-03-14", 3, 2],
    [2, "2016-03-14", None, 2],
    [3, "2016-03-14", None, 1],
    [4, "2016-03-14", 9, 1],
]


def test_available_imputations_coverage():
    assert set(available_imputations.keys()) == set(
        list(imputation_values.keys()) + ["error"]
    )


@pytest.mark.parametrize(
    ("feat_list", "exp_imp_cols", "feat_table"),
    [
        (["f1", "f2", "f3", "f4"], ["f1", "f2", "f3", "f4"], aggs_table),
        (["f5", "f6"], ["f5"], aggs_table_noimp),
    ],
)
def test_imputation_output(feat_list, exp_imp_cols, feat_table):
    with testing.postgresql.Postgresql() as psql:
        engine = sqlalchemy.create_engine(psql.url())

        engine.execute("create table states (entity_id int, as_of_date date)")
        for state in states_table:
            engine.execute("insert into states values (%s, %s)", state)

        feat_sql = "\n".join(
            [", prefix_entity_id_1y_%s_max int" % f for f in feat_list]
        )
        engine.execute(
            """create table prefix_aggregation (
                entity_id int
                , as_of_date date
                %s
                )"""
            % feat_sql
        )
        ins_sql = (
            "insert into prefix_aggregation values (%s, %s"
            + (", %s" * len(feat_list))
            + ")"
        )
        for rec in feat_table:
            engine.execute(ins_sql, rec)

        for imp in available_imputations.keys():
            # skip error imputation
            if imp == "error":
                continue

            for coltype in ["aggregate", "categorical"]:
                # only consider
                if not imputation_values[imp][coltype]["avail"]:
                    continue

                impargs = imputation_values[imp][coltype]["kwargs"]
                aggs = [
                    Aggregate(
                        feat,
                        ["max"],
                        {"coltype": coltype, "all": dict(type=imp, **impargs)},
                    )
                    for feat in feat_list
                ]
                st = SpacetimeAggregation(
                    aggregates=aggs,
                    from_obj="prefix_events",
                    prefix="prefix",
                    groups=["entity_id"],
                    intervals=["1y"],
                    dates=["2016-01-01", "2016-02-03", "2016-03-14"],
                    state_table="states",
                    state_group="entity_id",
                    date_column="as_of_date",
                    input_min_date="2000-01-01",
                    output_date_column="as_of_date",
                )

                conn = engine.connect()

                trans = conn.begin()

                # excute query to find columns with null values and create lists of columns
                # that do and do not need imputation when creating the imputation table
                res = conn.execute(st.find_nulls())
                null_counts = list(zip(res.keys(), res.fetchone()))
                impute_cols = [col for col, val in null_counts if val > 0]
                nonimpute_cols = [col for col, val in null_counts if val == 0]

                # sql to drop and create the imputation table
                drop_imp = st.get_drop(imputed=True)
                create_imp = st.get_impute_create(
                    impute_cols=impute_cols, nonimpute_cols=nonimpute_cols
                )

                # create the imputation table
                conn.execute(drop_imp)
                conn.execute(create_imp)

                trans.commit()

                # check the results
                df = pd.read_sql("SELECT * FROM prefix_aggregation_imputed", engine)

                # we should have a record for every entity/date combo
                assert df.shape[0] == len(states_table)

                for feat in feat_list:
                    # all of the input columns should be in the result and be null-free
                    assert "prefix_entity_id_1y_%s_max" % feat in df.columns.values
                    assert df["prefix_entity_id_1y_%s_max" % feat].isnull().sum() == 0

                    # for non-categoricals, should add an "imputed" column and be non-null
                    # (categoricals are expected to be handled through the null category)
                    # zero_noflag imputation should not generate a flag either
                    if (
                        feat in exp_imp_cols
                        and coltype != "categorical"
                        and imp != "zero_noflag"
                    ):
                        assert (
                            "prefix_entity_id_1y_%s_imp" % feat in df.columns.values
                        )
                        assert (
                            df["prefix_entity_id_1y_%s_imp" % feat].isnull().sum()
                            == 0
                        )
                    else:
                        # should not generate an imputed column when not needed
                        assert (
                            "prefix_entity_id_1y_%s_imp" % feat
                            not in df.columns.values
                        )
