import copy
from datetime import date

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import text as t

from triage.component.architect.feature_generators import FeatureGenerator
from triage.component.collate import Aggregate, Categorical, SpacetimeAggregation

from unittest.mock import patch


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


def test_feature_generation(test_engine):
    aggregate_config = [
        {
            "prefix": "aprefix",
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
    ]

    expected_output = {
        "aprefix_aggregation_imputed": [
            {
                "entity_id": 1,
                "as_of_date": date(2013, 9, 30),
                "zip_code": None,
                "aprefix_entity_id_all_quantity_one_sum": 137,
                "aprefix_entity_id_all_quantity_one_count": 0,
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 1,
                "aprefix_zip_code_all_quantity_one_sum": 137,
                "aprefix_zip_code_all_quantity_one_count": 0,
                "aprefix_zip_code_all_cat_one_good_sum": 0,
                "aprefix_zip_code_all_cat_one_bad_sum": 0,
                "aprefix_zip_code_all_cat_one__NULL_sum": 1,
                "aprefix_entity_id_all_quantity_one_imp": 1,
                "aprefix_zip_code_all_quantity_one_imp": 1,
            },
            {
                "entity_id": 1,
                "as_of_date": date(2014, 9, 30),
                "zip_code": "60120",
                "aprefix_entity_id_all_quantity_one_sum": 10000,
                "aprefix_entity_id_all_quantity_one_count": 1,
                "aprefix_entity_id_all_cat_one_good_sum": 1,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
                "aprefix_zip_code_all_quantity_one_sum": 10000,
                "aprefix_zip_code_all_quantity_one_count": 1,
                "aprefix_zip_code_all_cat_one_good_sum": 1,
                "aprefix_zip_code_all_cat_one_bad_sum": 0,
                "aprefix_zip_code_all_cat_one__NULL_sum": 0,
                "aprefix_entity_id_all_quantity_one_imp": 0,
                "aprefix_zip_code_all_quantity_one_imp": 0,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2013, 9, 30),
                "zip_code": "60653",
                "aprefix_entity_id_all_quantity_one_sum": 342,
                "aprefix_entity_id_all_quantity_one_count": 1,
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
                "aprefix_zip_code_all_quantity_one_sum": 342,
                "aprefix_zip_code_all_quantity_one_count": 1,
                "aprefix_zip_code_all_cat_one_good_sum": 0,
                "aprefix_zip_code_all_cat_one_bad_sum": 1,
                "aprefix_zip_code_all_cat_one__NULL_sum": 0,
                "aprefix_entity_id_all_quantity_one_imp": 0,
                "aprefix_zip_code_all_quantity_one_imp": 0,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2014, 9, 30),
                "zip_code": "60653",
                "aprefix_entity_id_all_quantity_one_sum": 342,
                "aprefix_entity_id_all_quantity_one_count": 1,
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
                "aprefix_zip_code_all_quantity_one_sum": 1578,
                "aprefix_zip_code_all_quantity_one_count": 2,
                "aprefix_zip_code_all_cat_one_good_sum": 0,
                "aprefix_zip_code_all_cat_one_bad_sum": 2,
                "aprefix_zip_code_all_cat_one__NULL_sum": 0,
                "aprefix_entity_id_all_quantity_one_imp": 0,
                "aprefix_zip_code_all_quantity_one_imp": 0,
            },
            {
                "entity_id": 4,
                "as_of_date": date(2014, 9, 30),
                "zip_code": "60653",
                "aprefix_entity_id_all_quantity_one_sum": 1236,
                "aprefix_entity_id_all_quantity_one_count": 1,
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
                "aprefix_zip_code_all_quantity_one_sum": 1578,
                "aprefix_zip_code_all_quantity_one_count": 2,
                "aprefix_zip_code_all_cat_one_good_sum": 0,
                "aprefix_zip_code_all_cat_one_bad_sum": 2,
                "aprefix_zip_code_all_cat_one__NULL_sum": 0,
                "aprefix_entity_id_all_quantity_one_imp": 0,
                "aprefix_zip_code_all_quantity_one_imp": 0,
            },
        ]
    }

    features_schema_name = "features"

    output_tables = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).create_all_tables(
        feature_dates=["2013-09-30", "2014-09-30"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )

    for output_table in output_tables:
        records = pd.read_sql(
            "select * from {}.{} order by entity_id, as_of_date".format(
                features_schema_name,
                output_table,
            ),
            test_engine,
        ).to_dict("records")

        for record, expected_record in zip(records, expected_output[output_table]):
            assert record == expected_record


def test_index_column_lookup(test_engine):
    aggregations = [
        SpacetimeAggregation(
            prefix="prefix1",
            aggregates=[
                Categorical(
                    col="cat_one",
                    function="sum",
                    choices=["good", "bad", "inbetween"],
                    impute_rules={"coltype": "categorical", "all": {"type": "zero"}},
                )
            ],
            groups=["entity_id"],
            intervals=["all"],
            date_column="knowledge_date",
            output_date_column="as_of_date",
            dates=["2013-09-30", "2014-09-30"],
            state_table="states",
            state_group="entity_id",
            schema="features",
            from_obj="data",
        ),
        SpacetimeAggregation(
            prefix="prefix2",
            aggregates=[
                Aggregate(
                    quantity="quantity_one",
                    function="count",
                    impute_rules={"coltype": "aggregate", "all": {"type": "zero"}},
                )
            ],
            groups=["entity_id", "zip_code"],
            intervals=["all"],
            date_column="knowledge_date",
            output_date_column="as_of_date",
            dates=["2013-09-30", "2014-09-30"],
            state_table="states",
            state_group="entity_id",
            schema="features",
            from_obj="data",
        ),
    ]

    features_schema_name = "features"
    feature_generator = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    )
    lookup = feature_generator.index_column_lookup(aggregations)
    assert lookup == {
        "prefix1_aggregation_imputed": ["as_of_date", "entity_id"],
        "prefix2_aggregation_imputed": ["as_of_date", "entity_id", "zip_code"],
    }


def test_feature_generation_feature_start_time(test_engine):
    aggregate_config = [
        {
            "prefix": "aprefix",
            "aggregates_imputation": {"all": {"type": "constant", "value": 7}},
            "aggregates": [{"quantity": "quantity_one", "metrics": ["sum"]}],
            "groups": ["entity_id"],
            "intervals": ["all"],
            "knowledge_date_column": "knowledge_date",
            "from_obj": "data",
        }
    ]

    expected_output = {
        "aprefix_aggregation_imputed": [
            {
                "entity_id": 1,
                "as_of_date": date(2015, 1, 1),
                "aprefix_entity_id_all_quantity_one_sum": 10000,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2015, 1, 1),
                "aprefix_entity_id_all_quantity_one_sum": 600,
            },
            {
                "entity_id": 4,
                "as_of_date": date(2015, 1, 1),
                "aprefix_entity_id_all_quantity_one_sum": 1236,
            },
        ]
    }

    features_schema_name = "features"
    output_tables = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
        feature_start_time="2013-01-01",
    ).create_all_tables(
        feature_dates=["2015-01-01"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )

    for output_table in output_tables:
        records = pd.read_sql(
            "select * from {}.{} order by as_of_date, entity_id".format(
                features_schema_name,
                output_table,
            ),
            test_engine,
        ).to_dict("records")

        assert records == expected_output[output_table]


def test_dynamic_categoricals(test_engine):
    aggregate_config = [
        {
            "prefix": "aprefix",
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
    ]
    expected_output = {
        "aprefix_aggregation_imputed": [
            {
                "entity_id": 1,
                "as_of_date": date(2013, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 1,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2013, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 1,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 1,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 4,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
        ]
    }

    features_schema_name = "features"

    output_tables = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).create_all_tables(
        feature_dates=["2013-09-30", "2014-09-30"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )

    for output_table in output_tables:
        records = pd.read_sql(
            "select * from {}.{} order by as_of_date, entity_id".format(
                features_schema_name, output_table
            ),
            test_engine,
        ).to_dict("records")

        assert records == expected_output[output_table]


def test_array_categoricals(db_engine):
    aggregate_config = [
        {
            "prefix": "aprefix",
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
    ]
    expected_output = {
        "aprefix_aggregation_imputed": [
            {
                "entity_id": 1,
                "as_of_date": date(2013, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 1,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2013, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 1,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 1,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 0,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 3,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
            {
                "entity_id": 4,
                "as_of_date": date(2014, 9, 30),
                "aprefix_entity_id_all_cat_one_good_sum": 0,
                "aprefix_entity_id_all_cat_one_inbetween_sum": 0,
                "aprefix_entity_id_all_cat_one_bad_sum": 1,
                "aprefix_entity_id_all_cat_one__NULL_sum": 0,
            },
        ]
    }

    input_data = [
        # entity_id, knowledge_date, cat_one, quantity_one
        (1, date(2014, 1, 1), ["good", "good"], 10000),
        (1, date(2014, 10, 11), ["good"], None),
        (3, date(2012, 6, 8), ["bad"], 342),
        (3, date(2014, 12, 21), ["inbetween"], 600),
        (4, date(2014, 4, 4), ["bad"], 1236),
    ]

    db_engine.execute(
        """\
        create table data (
            entity_id int,
            knowledge_date date,
            cat_one varchar[],
            quantity_one float
        )
        """
    )
    for row in input_data:
        db_engine.execute("insert into data values (%s, %s, %s, %s)", row)

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

    features_schema_name = "features"

    output_tables = FeatureGenerator(
        db_engine=db_engine,
        features_schema_name=features_schema_name,
    ).create_all_tables(
        feature_dates=["2013-09-30", "2014-09-30"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )

    for output_table in output_tables:
        records = pd.read_sql(
            "select * from {}.{} order by as_of_date, entity_id".format(
                features_schema_name, output_table
            ),
            db_engine,
        ).to_dict("records")

        assert records == expected_output[output_table]


def test_generate_table_tasks(test_engine):
    test_engine.execute('create schema features')
    aggregations = [
        SpacetimeAggregation(
            prefix="prefix1",
            aggregates=[
                Categorical(
                    col="cat_one",
                    function="sum",
                    choices=["good", "bad", "inbetween"],
                    impute_rules={"coltype": "categorical", "all": {"type": "zero"}},
                )
            ],
            groups=["entity_id"],
            intervals=["all"],
            date_column="knowledge_date",
            output_date_column="as_of_date",
            dates=["2013-09-30", "2014-09-30"],
            state_table="states",
            state_group="entity_id",
            schema="features",
            from_obj="data",
        ),
        SpacetimeAggregation(
            prefix="prefix2",
            aggregates=[
                Aggregate(
                    quantity="quantity_one",
                    function="count",
                    impute_rules={"coltype": "aggregate", "all": {"type": "zero"}},
                )
            ],
            groups=["entity_id"],
            intervals=["all"],
            date_column="knowledge_date",
            output_date_column="as_of_date",
            dates=["2013-09-30", "2014-09-30"],
            state_table="states",
            state_group="entity_id",
            schema="features",
            from_obj="data",
        ),
    ]
    features_schema_name = "features"

    table_tasks = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).generate_all_table_tasks(aggregations, task_type="aggregation")
    for table_name, task in table_tasks.items():
        assert "DROP TABLE" in task["prepare"][0]
        assert "CREATE TABLE" in str(task["prepare"][1])
        assert "CREATE INDEX" in task["finalize"][0]
        assert isinstance(task["inserts"], list)

    # build the aggregation tables to check the imputation tasks
    FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).process_table_tasks(table_tasks)

    table_tasks = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).generate_all_table_tasks(aggregations, task_type="imputation")

    for table_name, task in table_tasks.items():
        assert "DROP TABLE" in task["prepare"][0]
        assert "CREATE TABLE" in str(task["prepare"][1])
        assert "CREATE INDEX" in task["finalize"][0]
        assert isinstance(task["inserts"], list)


def test_aggregations(test_engine):
    aggregate_config = [
        {
            "prefix": "prefix1",
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
        },
        {
            "prefix": "prefix2",
            "aggregates_imputation": {"all": {"type": "mean"}},
            "aggregates": [{"quantity": "quantity_one", "metrics": ["count"]}],
            "groups": ["entity_id"],
            "intervals": ["all"],
            "knowledge_date_column": "knowledge_date",
            "from_obj": "data",
        },
    ]
    features_schema_name = "features"

    aggregations = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
    ).aggregations(
        feature_dates=["2013-09-30", "2014-09-30"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )
    for aggregation in aggregations:
        assert isinstance(aggregation, SpacetimeAggregation)


def test_replace(test_engine):
    # test the replace=False functionality, wherein we see if the cohort is fully represented
    # in the imputed table and reuse the features if so
    aggregate_config = [
        {
            "prefix": "aprefix",
            "aggregates_imputation": {"all": {"type": "mean"}},
            "aggregates": [{"quantity": "quantity_one", "metrics": ["sum", "count"]}],
            "categoricals": [
                {
                    "column": "cat_one",
                    "choices": ["good", "bad"],
                    "metrics": ["sum"],
                    "imputation": {"all": {"type": "null_category"}},
                }
            ],
            "groups": ["entity_id"],
            "intervals": ["all"],
            "knowledge_date_column": "knowledge_date",
            "from_obj": "data",
        }
    ]

    features_schema_name = "features"
    feature_tables = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
        replace=False,
    ).create_all_tables(
        feature_dates=["2013-09-30", "2014-09-30", "2015-01-01"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )

    assert len(feature_tables) == 1
    assert list(feature_tables)[0] == "aprefix_aggregation_imputed"

    # now try and run feature generation with replace=False. We should
    # be able to see that the entire cohort is there and reuse the features
    feature_generator = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name=features_schema_name,
        replace=False,
    )
    aggregations = feature_generator.aggregations(
        feature_dates=["2013-09-30", "2014-09-30", "2015-01-01"],
        feature_aggregation_config=aggregate_config,
        state_table="states",
    )
    table_tasks = feature_generator.generate_all_table_tasks(
        aggregations,
        task_type="aggregation",
    )

    assert len(table_tasks["aprefix_entity_id"]) == 0
    assert len(table_tasks["aprefix_aggregation"]) == 0

    imp_tasks = feature_generator.generate_all_table_tasks(
        aggregations,
        task_type="imputation",
    )

    assert len(imp_tasks["aprefix_aggregation_imputed"]) == 0

    # add a new member of the cohort. now we should need to rebuild everything
    test_engine.execute("insert into states values (%s, %s)", 999, "2015-01-01")
    table_tasks = feature_generator.generate_all_table_tasks(
        aggregations,
        task_type="aggregation",
    )
    assert len(table_tasks["aprefix_entity_id"]) == 3
    assert len(table_tasks["aprefix_aggregation"]) == 3
    feature_generator.process_table_tasks(table_tasks)
    imp_tasks = feature_generator.generate_all_table_tasks(
        aggregations,
        task_type="imputation",
    )

    assert len(imp_tasks["aprefix_aggregation_imputed"]) == 3

def test_aggregations_materialize_off(test_engine):
    aggregate_config = {
        "prefix": "aprefix",
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

    feature_generator = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name="features",
        materialize_subquery_fromobjs=False
    )

    with patch("triage.component.architect.feature_generators.FromObj") as fromobj_mock:
        feature_generator.aggregations([aggregate_config], "2016-01-01", "states")
        assert not fromobj_mock.called


def test_aggregations_materialize_on(test_engine):
    aggregate_config = {
        "prefix": "aprefix",
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

    feature_generator = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name="features",
    )

    with patch("triage.component.architect.feature_generators.FromObj") as fromobj_mock:
        feature_generator.aggregations([aggregate_config], "2016-01-01", "states")
        fromobj_mock.assert_called_once_with(
            from_obj="data",
            knowledge_date_column="knowledge_date",
            name="features.aprefix"
        )


def test_transaction_error(test_engine):
    """Database connections are cleaned up regardless of in-transaction
    query errors.

    """
    aggregate_config = [
        {
            "prefix": "aprefix",
            "aggregates": [
                {
                    "quantity": "quantity_one",
                    "metrics": ["sum"],
                    "imputation": {
                        "sum": {"type": "constant", "value": 137},
                        "count": {"type": "zero"},
                    },
                }
            ],
            "groups": ["entity_id"],
            "intervals": ["all"],
            "knowledge_date_column": "knowledge_date",
            "from_obj": "data",
        }
    ]

    feature_generator = FeatureGenerator(
        db_engine=test_engine,
        features_schema_name="features",
    )

    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        feature_generator.create_all_tables(
            feature_dates=["2013-09-30", "2014-09-30"],
            feature_aggregation_config=aggregate_config,
            state_table="statez",  # WRONG!
        )

    ((query_count,),) = test_engine.execute(
        t("""\
            select count(1) from pg_stat_activity
            where datname = :datname and
                  query not ilike '%%pg_stat_activity%%'
        """),
        datname=test_engine.url.database,
    )

    assert query_count == 0


class TestValidations:

    @pytest.fixture
    def base_config(self):
        return {
            "prefix": "aprefix",
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

    @pytest.fixture
    def feature_generator(self, test_engine):
        return FeatureGenerator(test_engine, "features")

    def test_correct_keys(self, base_config, feature_generator):
        feature_generator.validate([base_config])

        with pytest.raises(ValueError):
            no_group = copy.deepcopy(base_config)
            del no_group["groups"]
            feature_generator.validate([no_group])

        with pytest.raises(ValueError):
            no_intervals = copy.deepcopy(base_config)
            del no_intervals["intervals"]
            feature_generator.validate([no_intervals])

        with pytest.raises(ValueError):
            no_kdate = copy.deepcopy(base_config)
            del no_kdate["knowledge_date_column"]
            feature_generator.validate([no_kdate])

        with pytest.raises(ValueError):
            no_from_obj = copy.deepcopy(base_config)
            del no_from_obj["from_obj"]
            feature_generator.validate([no_from_obj])

        with pytest.raises(ValueError):
            no_aggs = copy.deepcopy(base_config)
            del no_aggs["categoricals"]
            feature_generator.validate([no_aggs])

        with pytest.raises(ValueError):
            no_imps = copy.deepcopy(base_config)
            del no_imps["categoricals"][0]["imputation"]
            feature_generator.validate([no_imps])

    def test_bad_from_obj(self, base_config, feature_generator):
        bad_from_obj = copy.deepcopy(base_config)
        bad_from_obj["from_obj"] = "where thing is other_thing"
        with pytest.raises(ValueError):
            feature_generator.validate([bad_from_obj])

    def test_bad_interval(self, base_config, feature_generator):
        base_config["intervals"] = ["1y", "1fortnight"]
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_bad_group(self, base_config, feature_generator):
        base_config["groups"] = ["zip_code", "otherthing"]
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_bad_choice_query(self, base_config, feature_generator):
        del base_config["categoricals"][0]["choices"]
        base_config["categoricals"][0][
            "choice_query"
        ] = "select distinct cat_two from data"
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_wrong_imp_fcn(self, base_config, feature_generator):
        del base_config["categoricals"][0]["imputation"]["all"]
        base_config["categoricals"][0]["imputation"]["max"] = {
            "type": "null_category"
        }
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_bad_imp_rule(self, base_config, feature_generator):
        base_config["categoricals"][0]["imputation"]["all"] = {
            "type": "bad_rule_doesnt_exist"
        }
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_no_imp_rule_type(self, base_config, feature_generator):
        base_config["categoricals"][0]["imputation"]["all"] = {"value": "good"}
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])

    def test_missing_imp_arg(self, base_config, feature_generator):
        # constant value imputation requires a 'value' parameter
        base_config["categoricals"][0]["imputation"]["all"] = {"type": "constant"}
        with pytest.raises(ValueError):
            feature_generator.validate([base_config])
