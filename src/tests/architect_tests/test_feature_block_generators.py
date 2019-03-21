from datetime import datetime, date

from triage.component.architect.feature_block_generators import generate_spacetime_aggregation
import triage.component.collate as collate

import pytest
from unittest.mock import patch


def test_spacetime_generation(db_engine):
    aggregation_config = {
        "aggregates": [
            {
                "quantity": "quantity_one",
                "metrics": ["sum", "count"],
                "imputation": {
                    "sum": {"type": "constant", "value": 137},
                    "count": {"type": "zero"},
                },
            }
        ],
        "categoricals_imputation": {"all": {"type": "null_category"}},
        "categoricals": [
            {"column": "cat_one", "choices": ["good", "bad"], "metrics": ["sum"]}
        ],
        "groups": ["entity_id", "zip_code"],
        "intervals": ["all"],
        "knowledge_date_column": "knowledge_date",
        "from_obj": "data",
    }
    aggregation = generate_spacetime_aggregation(
        feature_aggregation_config=aggregation_config,
        as_of_dates=["2017-01-02", "2017-02-02"],
        cohort_table="my_cohort",
        feature_table_name="my_features",
        db_engine=db_engine,
        features_schema_name="features",
        feature_start_time="2011-01-01",
    )
    assert isinstance(aggregation, collate.SpacetimeAggregation)
    assert aggregation.as_of_dates == ["2017-01-02", "2017-02-02"]
    assert aggregation.feature_start_time == "2011-01-01"
    assert aggregation.groups == {"entity_id": "entity_id", "zip_code": "zip_code"}
    assert aggregation.intervals == {"entity_id": ["all"], "zip_code": ["all"]}
    assert str(aggregation.from_obj) == "data"
    assert len(aggregation.aggregates) == 2
    for aggregate in aggregation.aggregates:
        if isinstance(aggregate, collate.Categorical):
            assert aggregate.quantities == {
                "cat_one__NULL": ('(cat_one is NULL)::INT',),
                "cat_one_bad": ("(cat_one = 'bad')::INT",),
                "cat_one_good": ("(cat_one = 'good')::INT",),
            }
            assert aggregate.functions == ["sum"]
        else:
            assert aggregate.quantities == {"quantity_one": ("quantity_one",)}
            assert aggregate.functions == ["sum", "count"]



INPUT_DATA = [
    # entity_id, knowledge_date, zip_code, cat_one, quantity_one
    (1, date(2014, 1, 1), "60120", "good", 10000),
    (1, date(2014, 10, 11), "60120", "good", None),
    (3, date(2012, 6, 8), "60653", "bad", 342),
    (3, date(2014, 12, 21), "60653", "inbetween", 600),
    (4, date(2014, 4, 4), "60653", "bad", 1236),
]

INPUT_STATES = [
    # entity_id, as_of_date
    (1, date(2013, 9, 30)),
    (1, date(2014, 9, 30)),
    (1, date(2015, 1, 1)),
    (3, date(2013, 9, 30)),
    (3, date(2014, 9, 30)),
    (3, date(2015, 1, 1)),
    (4, date(2014, 9, 30)),
    (4, date(2015, 1, 1)),
]

@pytest.fixture(name='test_engine', scope='function')
def fixture_test_engine(db_engine):
    """Local extension to the shared db_engine fixture to set up test
    database tables.

    """
    db_engine.execute(
        """\
        create table data (
            entity_id int,
            knowledge_date date,
            zip_code text,
            cat_one varchar,
            quantity_one float
        )
        """
    )
    for row in INPUT_DATA:
        db_engine.execute("insert into data values (%s, %s, %s, %s, %s)", row)

    db_engine.execute(
        """\
        create table states (
            entity_id int,
            as_of_date date
        )
        """
    )
    for row in INPUT_STATES:
        db_engine.execute("insert into states values (%s, %s)", row)

    return db_engine


def test_choice_query(test_engine):
    aggregation_config = {
        "categoricals": [
            {
                "column": "cat_one",
                "choice_query": "select distinct(cat_one) from data",
                "metrics": ["sum"],
                "imputation": {"all": {"type": "null_category"}},
            }
        ],
        "groups": ["entity_id"],
        "intervals": ["all"],
        "knowledge_date_column": "knowledge_date",
        "from_obj": "data",
    }
    aggregation = generate_spacetime_aggregation(
        feature_aggregation_config=aggregation_config,
        as_of_dates=["2017-01-02", "2017-02-02"],
        cohort_table="my_cohort",
        db_engine=test_engine,
        features_schema_name="features",
        feature_start_time="2011-01-01",
        feature_table_name="aprefix",
    )
    assert aggregation.aggregates[0].quantities == {
        "cat_one__NULL": ('(cat_one is NULL)::INT',),
        "cat_one_bad": ("(cat_one = 'bad')::INT",),
        "cat_one_good": ("(cat_one = 'good')::INT",),
        "cat_one_inbetween": ("(cat_one = 'inbetween')::INT",),
    }

def test_array_categoricals(test_engine):
    aggregation_config = {
        "array_categoricals": [
            {
                "column": "cat_one",
                "choices": ["good", "bad", "inbetween"],
                "metrics": ["sum"],
                "imputation": {"all": {"type": "null_category"}},
            }
        ],
        "groups": ["entity_id"],
        "intervals": ["all"],
        "knowledge_date_column": "knowledge_date",
        "from_obj": "data",
    }
    aggregation = generate_spacetime_aggregation(
        feature_aggregation_config=aggregation_config,
        as_of_dates=["2017-01-02", "2017-02-02"],
        cohort_table="my_cohort",
        db_engine=test_engine,
        features_schema_name="features",
        feature_start_time="2011-01-01",
        feature_table_name="aprefix",
    )

    assert aggregation.aggregates[0].quantities == {
        "cat_one__NULL": ('(cat_one is NULL)::INT',),
        "cat_one_bad": ("(cat_one @> array['bad'::varchar])::INT",),
        "cat_one_good": ("(cat_one @> array['good'::varchar])::INT",),
        "cat_one_inbetween": ("(cat_one @> array['inbetween'::varchar])::INT",),
    }

def xtest_materialize_off(db_engine):
    aggregation_config = {
        "categoricals": [
            {
                "column": "cat_one",
                "choices": ["good", "bad"],
                "metrics": ["sum"],
                "imputation": {"all": {"type": "null_category"}},
            }
        ],
        "groups": ["entity_id", "zip_code"],
        "intervals": ["all"],
        "knowledge_date_column": "knowledge_date",
        "from_obj": "data",
    }

    with patch("triage.component.architect.feature_block_generators.FromObj") as fromobj_mock:
        feature_generator = generate_spacetime_aggregation(
            feature_aggregation_config=aggregation_config,
            as_of_dates=["2017-01-02", "2017-02-02"],
            cohort_table="my_cohort",
            db_engine=db_engine,
            features_schema_name="features",
            materialize_subquery_fromobjs=False,
            feature_table_name="aprefix",
        )
        assert not fromobj_mock.called


def xtest_aggregations_materialize_on(db_engine):
    aggregation_config = {
        "categoricals": [
            {
                "column": "cat_one",
                "choices": ["good", "bad"],
                "metrics": ["sum"],
                "imputation": {"all": {"type": "null_category"}},
            }
        ],
        "groups": ["entity_id", "zip_code"],
        "intervals": ["all"],
        "knowledge_date_column": "knowledge_date",
        "from_obj": "data",
    }

    with patch("triage.component.architect.feature_block_generators.FromObj") as fromobj_mock:
        feature_generator = generate_spacetime_aggregation(
            feature_aggregation_config=aggregation_config,
            as_of_dates=["2017-01-02", "2017-02-02"],
            cohort_table="my_cohort",
            db_engine=db_engine,
            features_schema_name="features",
            materialize_subquery_fromobjs=True,
            feature_table_name="aprefix",
        )
        fromobj_mock.assert_called_once_with(
            from_obj="data",
            knowledge_date_column="knowledge_date",
            name="features.aprefix"
        )
