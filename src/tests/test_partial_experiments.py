import os
from os.path import isfile, join
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from tests.utils import open_side_effect, populate_source_data, sample_config
from triage import create_engine
from triage.database_reflection import schema_tables
from triage.experiments import SingleThreadedExperiment
from triage.validation_primitives import table_should_have_data


@pytest.fixture
def experiment_engine(db_engine):
    """Fixture that provides a db_engine with source data populated."""
    populate_source_data(db_engine)
    return db_engine


class TestGetSplits:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                assert experiment.split_definitions

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                with pytest.raises(ValueError):
                    experiment.validate()


class TestCohort:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "cohort_config": sample_config()["cohort_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                table_should_have_data(
                    experiment.cohort_table_name, experiment.db_engine
                )

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                with pytest.raises(ValueError):
                    experiment.validate()


class TestLabels:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "label_config": sample_config()["label_config"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                table_should_have_data(
                    experiment.labels_table_name, experiment.db_engine
                )

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                with pytest.raises(ValueError):
                    experiment.validate()


class TestPreimputationFeatures:
    config = {
        "temporal_config": sample_config()["temporal_config"],
        "feature_aggregations": sample_config()["feature_aggregations"],
        "config_version": sample_config()["config_version"],
        "random_seed": sample_config()["random_seed"],
    }

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                generated_tables = [
                    table
                    for table in schema_tables(
                        experiment.features_schema_name, experiment.db_engine
                    ).keys()
                    if "_aggregation" in table
                ]

                assert len(generated_tables) == len(
                    sample_config()["feature_aggregations"]
                )
                for table in generated_tables:
                    table_should_have_data(table, experiment.db_engine)

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
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

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                generated_tables = [
                    table
                    for table in schema_tables(
                        experiment.features_schema_name, experiment.db_engine
                    ).keys()
                    if "_aggregation_imputed" in table
                ]

                assert len(generated_tables) == len(
                    sample_config()["feature_aggregations"]
                )
                for table in generated_tables:
                    table_should_have_data(table, experiment.db_engine)

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
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

    def test_run(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.run()
                matrices_path = join(experiment.project_path, "matrices")
                matrices_and_metadata = [
                    f
                    for f in os.listdir(matrices_path)
                    if isfile(join(matrices_path, f))
                ]
                matrices = experiment.matrix_build_tasks
                assert len(matrices) > 0
                for matrix in matrices:
                    assert "{}.csv.gz".format(matrix) in matrices_and_metadata
                    assert "{}.yaml".format(matrix) in matrices_and_metadata

    def test_validate_nonstrict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                experiment.validate(strict=False)

    def test_validate_strict(self, experiment_engine):
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "triage.util.conf.open", side_effect=open_side_effect
            ) as mock_file:
                experiment = SingleThreadedExperiment(
                    config=self.config,
                    db_engine=experiment_engine,
                    project_path=os.path.join(temp_dir, "inspections"),
                    cleanup=False,
                    partial_run=True,
                )
                with pytest.raises(ValueError):
                    experiment.validate()
