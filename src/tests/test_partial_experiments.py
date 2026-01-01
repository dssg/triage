import os
from os.path import isfile, join
from tempfile import TemporaryDirectory

import pytest
from triage import create_engine

from tests.utils import sample_config, populate_source_data, open_side_effect

from triage.experiments import SingleThreadedExperiment
from triage.database_reflection import schema_tables
from triage.validation_primitives import table_should_have_data
from unittest import mock
from contextlib import contextmanager


@contextmanager
def prepare_experiment(config, db_engine):
    populate_source_data(db_engine)
    with TemporaryDirectory() as temp_dir:
        with mock.patch(
            "triage.util.conf.open", side_effect=open_side_effect
        ) as mock_file:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                project_path=os.path.join(temp_dir, "inspections"),
                cleanup=False,
                partial_run=True,
            )
            yield experiment


class TestGetSplits:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            assert experiment.split_definitions

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()


class TestCohort:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "cohort_config": sample_config()["cohort_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            table_should_have_data(experiment.cohort_table_name, experiment.db_engine)

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()


class TestLabels:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "label_config": sample_config()["label_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            table_should_have_data(experiment.labels_table_name, experiment.db_engine)

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()


class TestPreimputationFeatures:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            generated_tables = [
                table
                for table in schema_tables(
                    experiment.features_schema_name, experiment.db_engine
                ).keys()
                if "_aggregation" in table
            ]

            assert len(generated_tables) == len(sample_config()["feature_aggregations"])
            for table in generated_tables:
                table_should_have_data(table, experiment.db_engine)

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()


class TestPostimputationFeatures:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "cohort_config": sample_config()["cohort_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            generated_tables = [
                table
                for table in schema_tables(
                    experiment.features_schema_name, experiment.db_engine
                ).keys()
                if "_aggregation_imputed" in table
            ]

            assert len(generated_tables) == len(sample_config()["feature_aggregations"])
            for table in generated_tables:
                table_should_have_data(table, experiment.db_engine)

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()


class TestMatrices:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "cohort_config": sample_config()["cohort_config"],
        "label_config": sample_config()["label_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.run()
            matrices_path = join(experiment.project_path, "matrices")
            matrices_and_metadata = [
                f for f in os.listdir(matrices_path) if isfile(join(matrices_path, f))
            ]
            matrices = experiment.matrix_build_tasks
            assert len(matrices) > 0
            for matrix in matrices:
                assert "{}.csv.gz".format(matrix) in matrices_and_metadata
                assert "{}.yaml".format(matrix) in matrices_and_metadata

    def test_validate_nonstrict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self, db_engine):
        with prepare_experiment(self.config, db_engine) as experiment:
            with pytest.raises(ValueError):
                experiment.validate()
