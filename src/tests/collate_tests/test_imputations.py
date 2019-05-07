# -*- coding: utf-8 -*-
"""test_imputations

Unit tests for `collate.imputations` module.

"""

from triage.component.collate.imputations import (
    BaseImputation,
    ImputeMean,
    ImputeConstant,
    ImputeZero,
    ImputeZeroNoFlag,
    ImputeNullCategory,
    ImputeBinaryMode,
    ImputeError
)


def test_impute_flag():
    imp = BaseImputation(column="a", coltype="aggregate")
    assert (
        imp.imputed_flag_select_and_alias() == (
            'CASE WHEN "a" IS NULL THEN 1::SMALLINT ELSE 0::SMALLINT END',
            'a_imp'
        )
    )


def test_impute_flag_categorical():
    imp = BaseImputation(column="a", coltype="categorical")
    assert imp.imputed_flag_select_and_alias() == (None, None)


def test_mean_imputation():
    imp = ImputeMean(column="a", coltype="aggregate")
    assert imp.to_sql() == 'COALESCE("a", AVG("a") OVER ()::REAL, 0::REAL) AS "a" '

    imp = ImputeMean(column="a", coltype="aggregate", partitionby="date")
    assert imp.to_sql() == 'COALESCE("a", AVG("a") OVER (PARTITION BY date)::REAL, 0::REAL) AS "a" '

    imp = ImputeMean(column="a", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a", AVG("a") OVER ()::REAL, 0::REAL) AS "a" '

    imp = ImputeMean(column="a", coltype="aggregate", partitionby="date")
    assert imp.to_sql() == 'COALESCE("a", AVG("a") OVER (PARTITION BY date)::REAL, 0::REAL) AS "a" '

    imp = ImputeMean(column="a__NULL_mean", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a__NULL_mean", 1) AS "a__NULL_mean" '


def test_constant_imputation():
    imp = ImputeConstant(column="a", coltype="aggregate", value=3.14)
    assert imp.to_sql() == 'COALESCE("a", 3.14) AS "a" '

    imp = ImputeConstant(column="a_myval_max", coltype="categorical", value="myval")
    assert imp.to_sql() == 'COALESCE("a_myval_max", 1::SMALLINT) AS "a_myval_max" '

    imp = ImputeConstant(column="a_otherval_max", coltype="categorical", value="myval")
    assert imp.to_sql() == 'COALESCE("a_otherval_max", 0::SMALLINT) AS "a_otherval_max" '

    imp = ImputeConstant(column="a__NULL_mean", coltype="categorical", value="myval")
    assert imp.to_sql() == 'COALESCE("a__NULL_mean", 1::SMALLINT) AS "a__NULL_mean" '


def test_impute_zero():
    imp = ImputeZero(column="a", coltype="aggregate")
    assert imp.to_sql() == 'COALESCE("a", 0::REAL) AS "a" '

    imp = ImputeZero(column="a_myval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_myval_max", 0::SMALLINT) AS "a_myval_max" '

    imp = ImputeZero(column="a_otherval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_otherval_max", 0::SMALLINT) AS "a_otherval_max" '

    imp = ImputeZero(column="a__NULL_mean", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a__NULL_mean", 1::SMALLINT) AS "a__NULL_mean" '


def test_impute_zero_noflag():
    imp = ImputeZeroNoFlag(column="a", coltype="aggregate")
    assert imp.to_sql() == 'COALESCE("a", 0::SMALLINT) AS "a" '
    assert imp.imputed_flag_select_and_alias() == (None, None)
    assert imp.noflag

    imp = ImputeZeroNoFlag(column="a_myval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_myval_max", 0::SMALLINT) AS "a_myval_max" '

    imp = ImputeZeroNoFlag(column="a_otherval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_otherval_max", 0::SMALLINT) AS "a_otherval_max" '

    imp = ImputeZeroNoFlag(column="a__NULL_mean", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a__NULL_mean", 0::SMALLINT) AS "a__NULL_mean" '


def test_impute_null_cat():
    imp = ImputeNullCategory(column="a_myval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_myval_max", 0::SMALLINT) AS "a_myval_max" '

    imp = ImputeNullCategory(column="a_otherval_max", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a_otherval_max", 0::SMALLINT) AS "a_otherval_max" '

    imp = ImputeNullCategory(column="a__NULL_mean", coltype="categorical")
    assert imp.to_sql() == 'COALESCE("a__NULL_mean", 1::SMALLINT) AS "a__NULL_mean" '

    try:
        imp = ImputeNullCategory(column="a", coltype="aggregate")
        imp.to_sql()
        assert False
    except ValueError:
        assert True
    else:
        assert False


def test_impute_binary_mode():
    imp = ImputeBinaryMode(column="a", coltype="aggregate")
    assert (
        imp.to_sql()
        == 'COALESCE("a", CASE WHEN AVG("a") OVER () > 0.5 THEN 1 ELSE 0 END, 0) AS "a" '
    )

    imp = ImputeBinaryMode(column="a", coltype="aggregate", partitionby="date")
    assert (
        imp.to_sql()
        == 'COALESCE("a", CASE WHEN AVG("a") OVER (PARTITION BY date) > 0.5 '
        'THEN 1 ELSE 0 END, 0) AS "a" '
    )

    try:
        imp = ImputeBinaryMode(column="a", coltype="categorical")
        imp.to_sql()
        assert False
    except ValueError:
        assert True
    else:
        assert False


def test_impute_error():
    try:
        imp = ImputeError(column="a", coltype="aggregate")
        imp.to_sql()
        assert False
    except ValueError:
        assert True
    else:
        assert False
