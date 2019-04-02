from triage.component.architect.feature_block import FeatureBlock
import pytest


class FeatureBlockExample(FeatureBlock):
    """A sample, functional FeatureBlock class

    Implements very simple versions of all of the abstract methods/properties
    that allows testing of the concrete methods in the base class
    """
    @property
    def final_feature_table_name(self):
        return "myfeatures"

    @property
    def feature_columns(self):
        return set(["feature_one", "feature_two"])

    @property
    def preinsert_queries(self):
        return [
            "drop table if exists myfeatures",
            "create table myfeatures (entity_id int, as_of_date timestamp, f_one int, f_two int)"
        ]

    @property
    def insert_queries(self):
        return [
            "insert into myfeatures values (1, '2016-01-01', 1, 0)",
            "insert into myfeatures values (1, '2016-02-01', 0, 0)",
            "insert into myfeatures values (2, '2016-01-01', 0, 1)",
            "insert into myfeatures values (2, '2016-02-01', 0, NULL)"
        ]

    @property
    def postinsert_queries(self):
        return [
            "create index on myfeatures (as_of_date)"
        ]

    @property
    def imputation_queries(self):
        return [
            "update myfeatures set f_one = 1 where f_one is null",
            "update myfeatures set f_two = 1 where f_two is null",
        ]


def populate_cohort(db_engine):
    db_engine.execute("create table mycohort (entity_id int, as_of_date timestamp)")
    db_engine.execute("insert into mycohort values (1, '2016-01-01'), "
                      "(1, '2016-02-01'), (2, '2016-01-01'), (2, '2016-02-01')")


def test_FeatureBlock_generate_preimpute_tasks(db_engine):
    block = FeatureBlockExample(
        db_engine=db_engine,
        cohort_table="mycohort",
        features_table_name="myfeaturetable",
        as_of_dates=['2016-01-01', '2016-02-01']
    )
    block.needs_features = lambda: True
    assert block.generate_preimpute_tasks(replace=False) == {
        "prepare": block.preinsert_queries,
        "inserts": block.insert_queries,
        "finalize": block.postinsert_queries
    }
    block.needs_features = lambda: False
    assert block.generate_preimpute_tasks(replace=False) == {}

    assert block.generate_preimpute_tasks(replace=True) == {
        "prepare": block.preinsert_queries,
        "inserts": block.insert_queries,
        "finalize": block.postinsert_queries
    }


def test_FeatureBlock_generate_impute_tasks(db_engine):
    block = FeatureBlockExample(
        db_engine=db_engine,
        cohort_table="mycohort",
        features_table_name="myfeaturetable",
        as_of_dates=['2016-01-01', '2016-02-01']
    )
    block.needs_features = lambda: True
    assert block.generate_impute_tasks(replace=False) == {
        "prepare": block.imputation_queries,
        "inserts": [],
        "finalize": []
    }
    block.needs_features = lambda: False
    assert block.generate_impute_tasks(replace=False) == {}

    assert block.generate_impute_tasks(replace=True) == {
        "prepare": block.imputation_queries,
        "inserts": [],
        "finalize": []
    }


def test_FeatureBlock_log_verbose_task_info(db_engine):
    block = FeatureBlockExample(
        db_engine=db_engine,
        cohort_table="mycohort",
        features_table_name="myfeaturetable",
        as_of_dates=['2016-01-01', '2016-02-01']
    )
    task = block.generate_impute_tasks(replace=True)
    # just want to make sure that the logging doesn't error, no assertions
    block.log_verbose_task_info(task)


def test_FeatureBlock_needs_features(db_engine):
    # needs_features should function as following:
    # if there are members of the cohort without features, needs_features should return true
    # 1. a freshly created table should definitely need features
    block = FeatureBlockExample(
        db_engine=db_engine,
        cohort_table="mycohort",
        features_table_name="myfeaturetable",
        as_of_dates=['2016-01-01', '2016-02-01']
    )
    populate_cohort(db_engine)
    assert block.needs_features()
    block.run_preimputation()
    block.run_imputation()
    assert not block.needs_features()

    # 2. a table that already has features, but is merely a subset of the cohort,
    # should also need features
    db_engine.execute("insert into mycohort values (3, '2016-01-01')")
    assert block.needs_features()


def test_FeatureBlock_verify_nonulls(db_engine):
    # verify_no_nulls should function as following:
    # if there are members of the cohort without features, needs_features should return true
    # 1. a freshly created table should definitely need features
    block = FeatureBlockExample(
        db_engine=db_engine,
        cohort_table="mycohort",
        features_table_name="myfeaturetable",
        as_of_dates=['2016-01-01', '2016-02-01']
    )
    populate_cohort(db_engine)
    block.run_preimputation()
    with pytest.raises(ValueError):
        block.verify_no_nulls()
    block.run_imputation()
    block.verify_no_nulls()
