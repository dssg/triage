import os
from os.path import isfile, join
from tempfile import TemporaryDirectory

import testing.postgresql
from triage import create_engine

from tests.utils import sample_config, populate_source_data, open_side_effect

from triage.experiments import SingleThreadedExperiment
from triage.database_reflection import schema_tables
from triage.validation_primitives import table_should_have_data
from unittest import TestCase, mock
from contextlib import contextmanager


@contextmanager
def prepare_experiment(config):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
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


class GetSplits(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
            experiment.run()
            assert experiment.split_definitions

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()


class Cohort(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "cohort_config": sample_config()["cohort_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
            experiment.run()
            table_should_have_data(experiment.cohort_table_name, experiment.db_engine)

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()


class Labels(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "label_config": sample_config()["label_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
            experiment.run()
            table_should_have_data(experiment.labels_table_name, experiment.db_engine)

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()


class PreimputationFeatures(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
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

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()


class PostimputationFeatures(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "cohort_config": sample_config()["cohort_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
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

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()


class Matrices(TestCase):
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "cohort_config": sample_config()["cohort_config"],
        "label_config": sample_config()["label_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self):
        with prepare_experiment(self.config) as experiment:
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

    def test_validate_nonstrict(self):
        with prepare_experiment(self.config) as experiment:
            experiment.validate(strict=False)

    def test_validate_strict(self):
        with prepare_experiment(self.config) as experiment:
            with self.assertRaises(ValueError):
                experiment.validate()
