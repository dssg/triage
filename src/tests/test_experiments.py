import os
import time
from datetime import datetime, timedelta
from functools import partial
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from tests.utils import open_side_effect, populate_source_data, sample_config
from triage import create_engine
from triage.component.catwalk.storage import CSVMatrixStore
from triage.component.results_schema.schema import Experiment
from triage.experiments import (
    CONFIG_VERSION,
    MultiCoreExperiment,
    SingleThreadedExperiment,
)
from triage.logging import ic


def num_linked_evaluations(db_engine):
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                """
            select count(*) from test_results.evaluations e
            join triage_metadata.models using (model_id)
            join test_results.predictions p on (
                e.model_id = p.model_id and
                e.evaluation_start_time <= p.as_of_date and
                e.evaluation_end_time >= p.as_of_date)
        """
            )
        )
        ((count,),) = result
        return count


# MultiCoreExperiment tests are skipped in CI because multiprocessing
# with pytest-postgresql causes deadlocks - forked processes inherit
# corrupted database connection pools. These tests work locally but
# hang indefinitely in CI environments.
parametrize_experiment_classes = pytest.mark.parametrize(
    ("experiment_class",),
    [
        (SingleThreadedExperiment,),
        pytest.param(
            partial(MultiCoreExperiment, n_processes=2, n_db_processes=2),
            marks=pytest.mark.skip(reason="MultiCoreExperiment deadlocks with pytest-postgresql in CI"),
        ),
    ],
)


@parametrize_experiment_classes
def test_filepaths_and_queries_give_same_hashes(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        query_config = sample_config(query_source="query")
        file_config = sample_config(query_source="filepath")

        experiment_with_queries = experiment_class(
            config=query_config,
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        )
        experiment_with_filepaths = experiment_class(
            config=file_config,
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        )
        assert experiment_with_queries.experiment_hash == experiment_with_filepaths.experiment_hash
        assert experiment_with_queries.cohort_table_name == experiment_with_filepaths.cohort_table_name
        assert experiment_with_queries.labels_table_name == experiment_with_filepaths.labels_table_name


@parametrize_experiment_classes
def test_simple_experiment(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        ).run()

        with db_engine.connect() as conn:
            # assert
            # 1. that model groups entries are present
            num_mgs = len(list(conn.execute(text("select * from triage_metadata.model_groups"))))
            ic(f"========================Model groups {num_mgs}")
            assert num_mgs > 0

            # 2. that model entries are present, and linked to model groups
            num_models = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from triage_metadata.model_groups
                join triage_metadata.models using (model_group_id)
                where model_comment = 'test2-final-final'
            """
                        )
                    )
                )
            )
            ic(f"========================Model {num_models}")
            assert num_models > 0

            # 3. predictions, linked to models for both training and testing predictions
            for set_type in ("train", "test"):
                num_predictions = len(
                    list(
                        conn.execute(
                            text(
                                f"""
                    select * from {set_type}_results.predictions
                    join triage_metadata.models using (model_id)"""
                            )
                        )
                    )
                )
                ic(f"========================Predictions {num_predictions}")
                assert num_predictions > 0

            # 4. evaluations linked to predictions linked to models, for training and testing
            for set_type in ("train", "test"):
                num_evaluations = len(
                    list(
                        conn.execute(
                            text(
                                f"""
                    select * from {set_type}_results.evaluations e
                    join triage_metadata.models using (model_id)
                    join {set_type}_results.predictions p on (
                        e.model_id = p.model_id and
                        e.evaluation_start_time <= p.as_of_date and
                        e.evaluation_end_time >= p.as_of_date)
                """
                            )
                        )
                    )
                )
                ic(f"========================Evaluations {num_evaluations}")
                assert num_evaluations > 0

            # 5. subset evaluations linked to subsets and predictions linked to
            #    models, for training and testing
            for set_type in ("train", "test"):
                num_evaluations = len(
                    list(
                        conn.execute(
                            text(
                                f"""
                    select e.model_id, e.subset_hash from {set_type}_results.evaluations e
                    join triage_metadata.models using (model_id)
                    join triage_metadata.subsets using (subset_hash)
                    join {set_type}_results.predictions p on (
                        e.model_id = p.model_id and
                        e.evaluation_start_time <= p.as_of_date and
                        e.evaluation_end_time >= p.as_of_date)
                    group by e.model_id, e.subset_hash
                    """
                            )
                        )
                    )
                )
                # 4 model groups trained/tested on 2 splits, with 1 metric + parameter
                assert num_evaluations == 8

            # 6. experiment
            num_experiments = len(list(conn.execute(text("select * from triage_metadata.experiments"))))
            assert num_experiments == 1

            # 7. that models are linked to experiments
            num_models_with_experiment = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from triage_metadata.experiments
                join triage_metadata.experiment_models using (experiment_hash)
                join triage_metadata.models using (model_hash)
            """
                        )
                    )
                )
            )
            assert num_models == num_models_with_experiment

            # 8. that models have the train end date and label timespan
            results = [
                (row.train_end_time, row.training_label_timespan)
                for row in conn.execute(
                    text("select train_end_time, training_label_timespan from triage_metadata.models")
                )
            ]
            # sample_config uses training_label_timespans=12months and model_update_frequency=1year
            assert sorted(set(results)) == [
                (datetime(2012, 12, 1), timedelta(365)),
                (datetime(2013, 12, 1), timedelta(365)),
            ]

            # 9. that the right number of individual importances are present
            # Note: individual_importance is commented out in sample_config(),
            # so no individual importances are computed
            individual_importances = list(
                conn.execute(
                    text(
                        """
            select * from test_results.individual_importances
            join triage_metadata.models using (model_id)
        """
                    )
                )
            )
            assert len(individual_importances) == 0  # individual_importance not configured

            # 10. Checking the proper matrices created and stored
            matrices = list(
                conn.execute(
                    text(
                        """
            select matrix_type, num_observations from triage_metadata.matrices"""
                    )
                )
            )
            types = [i[0] for i in matrices]
            counts = [i[1] for i in matrices]
            assert types.count("train") == 2
            assert types.count("test") == 2
            for i in counts:
                assert i > 0
            assert len(matrices) == 4

            # 11. Checking that all matrices are associated with the experiment
            linked_matrices = list(
                conn.execute(
                    text(
                        """select * from triage_metadata.matrices
            join triage_metadata.experiment_matrices using (matrix_uuid)
            join triage_metadata.experiments using (experiment_hash)"""
                    )
                )
            )
            assert len(linked_matrices) == len(matrices)


@parametrize_experiment_classes
def test_validate_default(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        )
        experiment.validate = mock.MagicMock()
        experiment.run()
        experiment.validate.assert_called_once()


@parametrize_experiment_classes
def test_skip_validation(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
            skip_validation=True,
        )
        experiment.validate = mock.MagicMock()
        experiment.run()
        experiment.validate.assert_not_called()


@parametrize_experiment_classes
def test_restart_experiment(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        )
        experiment.run()

        evaluations = num_linked_evaluations(db_engine)
        assert evaluations > 0

        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
            replace=False,
        )
        experiment.make_entity_date_table = mock.Mock()
        experiment.run()
        assert not experiment.make_entity_date_table.called


class TestConfigVersion:
    def test_load_if_right_version(self, db_engine):
        experiment_config = sample_config()
        experiment_config["config_version"] = CONFIG_VERSION
        with (
            TemporaryDirectory() as temp_dir,
            mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
        ):
            experiment = SingleThreadedExperiment(
                config=experiment_config,
                db_engine=db_engine,
                project_path=os.path.join(temp_dir, "inspections"),
            )

        assert isinstance(experiment, SingleThreadedExperiment)

    def test_noload_if_wrong_version(self):
        experiment_config = sample_config()
        experiment_config["config_version"] = "v0"
        with (
            TemporaryDirectory() as temp_dir,
            mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
        ):
            with pytest.raises(ValueError):
                SingleThreadedExperiment(
                    config=experiment_config,
                    db_engine=None,
                    project_path=os.path.join(temp_dir, "inspections"),
                )


@parametrize_experiment_classes
@mock.patch(
    "triage.component.architect.entity_date_table_generators.EntityDateTableGenerator.clean_up",
    side_effect=lambda: time.sleep(1),
)
def test_cleanup_timeout(_clean_up_mock, experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
            cleanup_timeout=0.02,  # Set short timeout
        )
        with pytest.raises(TimeoutError):
            experiment()


@parametrize_experiment_classes
def test_build_error(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
            skip_validation=True,  # avoid catching the missing data at validation stage
        )

        with mock.patch.object(experiment, "generate_matrices") as build_mock:
            build_mock.side_effect = RuntimeError("boom!")

            with pytest.raises(RuntimeError):
                experiment()


@parametrize_experiment_classes
@mock.patch(
    "triage.component.architect.entity_date_table_generators.EntityDateTableGenerator.clean_up",
    side_effect=lambda: time.sleep(1),
)
def test_build_error_cleanup_timeout(_clean_up_mock, experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        experiment = experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
            cleanup_timeout=0.02,  # Set short timeout
            skip_validation=True,  # avoid catching the missing data at validation stage
        )

        with mock.patch.object(experiment, "generate_matrices") as build_mock:
            build_mock.side_effect = RuntimeError("boom!")

            with pytest.raises(TimeoutError) as exc_info:
                experiment()

    # Last exception is TimeoutError, but earlier error is preserved in
    # __context__, and will be noted as well in any standard traceback:
    assert exc_info.value.__context__ is build_mock.side_effect


@parametrize_experiment_classes
def test_custom_label_name(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        config = sample_config()
        config["label_config"]["name"] = "custom_label_name"
        experiment = experiment_class(
            config=config,
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
        )
        assert experiment.label_generator.label_name == "custom_label_name"
        assert experiment.planner.label_names == ["custom_label_name"]


def test_profiling(db_engine):
    populate_source_data(db_engine)
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        project_path = os.path.join(temp_dir, "inspections")
        SingleThreadedExperiment(
            config=sample_config(),
            db_engine=db_engine,
            project_path=project_path,
            profile=True,
        ).run()
        assert len(os.listdir(os.path.join(project_path, "profiling_stats"))) == 1


@parametrize_experiment_classes
def test_baselines_with_missing_features(experiment_class, db_engine):
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        populate_source_data(db_engine)

        # set up the config with the baseline model and feature group mixing
        config = sample_config()
        # Feature name uses "all" interval since sample_config was updated in 249af99e
        config["grid_config"] = {
            "triage.component.catwalk.baselines.rankers.PercentileRankOneFeature": {
                "feature": ["entity_features_entity_id_all_cat_sightings_count"]
            }
        }
        config["feature_group_definition"] = {
            "tables": [
                "entity_features_aggregation_imputed",
                "zip_code_features_aggregation_imputed",
            ]
        }
        config["feature_group_strategies"] = ["leave-one-in"]
        experiment_class(
            config=config,
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
        ).run()

        with db_engine.connect() as conn:
            # assert
            # 1. that model groups entries are present
            num_mgs = len(list(conn.execute(text("select * from triage_metadata.model_groups"))))
            assert num_mgs > 0

            # 2. that model entries are present, and linked to model groups
            num_models = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from triage_metadata.model_groups
                join triage_metadata.models using (model_group_id)
                where model_comment = 'test2-final-final'
            """
                        )
                    )
                )
            )
            assert num_models > 0

            # 3. predictions, linked to models
            num_predictions = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from test_results.predictions
                join triage_metadata.models using (model_id)"""
                        )
                    )
                )
            )
            assert num_predictions > 0

            # 4. evaluations linked to predictions linked to models
            num_evaluations = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from test_results.evaluations e
                join triage_metadata.models using (model_id)
                join test_results.predictions p on (
                    e.model_id = p.model_id and
                    e.evaluation_start_time <= p.as_of_date and
                    e.evaluation_end_time >= p.as_of_date)
            """
                        )
                    )
                )
            )
            assert num_evaluations > 0

            # 5. experiment
            num_experiments = len(list(conn.execute(text("select * from triage_metadata.experiments"))))
            assert num_experiments == 1

            # 6. that models are linked to experiments
            num_models_with_experiment = len(
                list(
                    conn.execute(
                        text(
                            """
                select * from triage_metadata.experiments
                join triage_metadata.experiment_models using (experiment_hash)
                join triage_metadata.models using (model_hash)
            """
                        )
                    )
                )
            )
            assert num_models == num_models_with_experiment

            # 7. that models have the train end date and label timespan
            results = [
                (row.train_end_time, row.training_label_timespan)
                for row in conn.execute(
                    text("select train_end_time, training_label_timespan from triage_metadata.models")
                )
            ]
            # sample_config uses training_label_timespans=12months and model_update_frequency=1year
            assert sorted(set(results)) == [
                (datetime(2012, 12, 1), timedelta(365)),
                (datetime(2013, 12, 1), timedelta(365)),
            ]

            # 8. that the right number of individual importances are present
            # Note: individual_importance is commented out in sample_config(),
            # so no individual importances are computed
            individual_importances = list(
                conn.execute(
                    text(
                        """
            select * from test_results.individual_importances
            join triage_metadata.models using (model_id)
        """
                    )
                )
            )
            assert len(individual_importances) == 0  # individual_importance not configured


def test_serializable_engine_check_sqlalchemy_fail(db_engine):
    """If we pass a vanilla sqlalchemy engine to the experiment we should blow up"""
    with (
        TemporaryDirectory() as temp_dir,
        mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file,
    ):
        # Create a vanilla sqlalchemy engine (not triage's serializable one)
        vanilla_engine = sqlalchemy.create_engine(db_engine.url)
        with pytest.raises(TypeError):
            MultiCoreExperiment(
                config=sample_config(),
                db_engine=vanilla_engine,
                project_path=os.path.join(temp_dir, "inspections"),
            )


def test_experiment_metadata(finished_experiment):
    session = sessionmaker(bind=finished_experiment.db_engine)()
    experiment_row = session.query(Experiment).get(finished_experiment.experiment_hash)
    assert experiment_row.time_splits == 2
    assert experiment_row.as_of_times == 722  # updated after sample_config changes in 249af99e
    assert experiment_row.feature_blocks == 2
    assert experiment_row.feature_group_combinations == 1
    assert (
        experiment_row.matrices_needed == experiment_row.time_splits * 2 * experiment_row.feature_group_combinations
    )  # x2 for train and test
    assert experiment_row.grid_size == 4
    assert (
        experiment_row.models_needed == (experiment_row.matrices_needed / 2) * experiment_row.grid_size
    )  # /2 because we only need models per train matrix
    session.close()
