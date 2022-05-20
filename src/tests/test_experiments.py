import os
import time
from datetime import datetime, timedelta
from functools import partial
from tempfile import TemporaryDirectory
from unittest import mock, TestCase
import fakeredis

import pytest
import testing.postgresql
from triage import create_engine
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from tests.utils import sample_config, populate_source_data, open_side_effect
from triage.component.catwalk.storage import CSVMatrixStore
from triage.component.results_schema.schema import Experiment

from triage.experiments import (
    MultiCoreExperiment,
    SingleThreadedExperiment,
    CONFIG_VERSION,
)

from triage.experiments.rq import RQExperiment


def num_linked_evaluations(db_engine):
    ((result,),) = db_engine.execute(
        """
        select count(*) from test_results.evaluations e
        join triage_metadata.models using (model_id)
        join test_results.predictions p on (
            e.model_id = p.model_id and
            e.evaluation_start_time <= p.as_of_date and
            e.evaluation_end_time >= p.as_of_date)
    """
    )
    return result


parametrize_experiment_classes = pytest.mark.parametrize(
    ("experiment_class",),
    [
        (SingleThreadedExperiment,),
        (partial(MultiCoreExperiment, n_processes=2, n_db_processes=2),),
        (
            partial(
                RQExperiment,
                redis_connection=fakeredis.FakeStrictRedis(),
                queue_kwargs={"is_async": False},
            ),
        ),
    ],
)


@parametrize_experiment_classes
def test_filepaths_and_queries_give_same_hashes(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
        assert (
            experiment_with_queries.experiment_hash
            == experiment_with_filepaths.experiment_hash
        )
        assert (
            experiment_with_queries.cohort_table_name
            == experiment_with_filepaths.cohort_table_name
        )
        assert (
            experiment_with_queries.labels_table_name
            == experiment_with_filepaths.labels_table_name
        )


@parametrize_experiment_classes
def test_simple_experiment(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
        populate_source_data(db_engine)
        experiment_class(
            config=sample_config(),
            db_engine=db_engine,
            project_path=os.path.join(temp_dir, "inspections"),
            cleanup=True,
        ).run()

        # assert
        # 1. that model groups entries are present
        num_mgs = len(
            [
                row
                for row in db_engine.execute(
                    "select * from triage_metadata.model_groups"
                )
            ]
        )
        print(f"========================Model groups {num_mgs}")
        assert num_mgs > 0

        # 2. that model entries are present, and linked to model groups
        num_models = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from triage_metadata.model_groups
                join triage_metadata.models using (model_group_id)
                where model_comment = 'test2-final-final'
            """
                )
            ]
        )
        print(f"========================Model {num_models}")
        assert num_models > 0

        # 3. predictions, linked to models for both training and testing predictions
        for set_type in ("train", "test"):
            num_predictions = len(
                [
                    row
                    for row in db_engine.execute(
                        """
                    select * from {}_results.predictions
                    join triage_metadata.models using (model_id)""".format(
                            set_type, set_type
                        )
                    )
                ]
            )
            print(f"========================Predictions {num_predictions}")
            assert num_predictions > 0

        # 4. evaluations linked to predictions linked to models, for training and testing

        for set_type in ("train", "test"):
            num_evaluations = len(
                [
                    row
                    for row in db_engine.execute(
                        """
                    select * from {}_results.evaluations e
                    join triage_metadata.models using (model_id)
                    join {}_results.predictions p on (
                        e.model_id = p.model_id and
                        e.evaluation_start_time <= p.as_of_date and
                        e.evaluation_end_time >= p.as_of_date)
                """.format(
                            set_type, set_type, set_type
                        )
                    )
                ]
            )
            print(f"========================Evaluations {num_evaluations}")
            assert num_evaluations > 0

        # 5. subset evaluations linked to subsets and predictions linked to
        #    models, for training and testing
        for set_type in ("train", "test"):
            num_evaluations = len(
                [
                    row
                    for row in db_engine.execute(
                        """
                        select e.model_id, e.subset_hash from {}_results.evaluations e
                        join triage_metadata.models using (model_id)
                        join triage_metadata.subsets using (subset_hash)
                        join {}_results.predictions p on (
                            e.model_id = p.model_id and
                            e.evaluation_start_time <= p.as_of_date and
                            e.evaluation_end_time >= p.as_of_date)
                        group by e.model_id, e.subset_hash
                        """.format(
                            set_type, set_type
                        )
                    )
                ]
            )
            # 4 model groups trained/tested on 2 splits, with 1 metric + parameter
            assert num_evaluations == 8

        # 6. experiment
        num_experiments = len(
            [
                row
                for row in db_engine.execute(
                    "select * from triage_metadata.experiments"
                )
            ]
        )
        assert num_experiments == 1

        # 7. that models are linked to experiments
        num_models_with_experiment = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from triage_metadata.experiments
                join triage_metadata.experiment_models using (experiment_hash)
                join triage_metadata.models using (model_hash)
            """
                )
            ]
        )
        assert num_models == num_models_with_experiment

        # 8. that models have the train end date and label timespan
        results = [
            (model["train_end_time"], model["training_label_timespan"])
            for model in db_engine.execute("select * from triage_metadata.models")
        ]
        assert sorted(set(results)) == [
            (datetime(2012, 6, 1), timedelta(180)),
            (datetime(2013, 6, 1), timedelta(180)),
        ]

        # 9. that the right number of individual importances are present
        individual_importances = [
            row
            for row in db_engine.execute(
                """
            select * from test_results.individual_importances
            join triage_metadata.models using (model_id)
        """
            )
        ]
        assert len(individual_importances) == num_predictions * 2  # only 2 features

        # 10. Checking the proper matrices created and stored
        matrices = [
            row
            for row in db_engine.execute(
                """
            select matrix_type, num_observations from triage_metadata.matrices"""
            )
        ]
        types = [i[0] for i in matrices]
        counts = [i[1] for i in matrices]
        assert types.count("train") == 2
        assert types.count("test") == 2
        for i in counts:
            assert i > 0
        assert len(matrices) == 4

        # 11. Checking that all matrices are associated with the experiment
        linked_matrices = list(
            db_engine.execute(
                """select * from triage_metadata.matrices
            join triage_metadata.experiment_matrices using (matrix_uuid)
            join triage_metadata.experiments using (experiment_hash)"""
            )
        )
        assert len(linked_matrices) == len(matrices)


@parametrize_experiment_classes
def test_validate_default(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
def test_skip_validation(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
def test_restart_experiment(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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


class TestConfigVersion(TestCase):
    def test_load_if_right_version(self):
        experiment_config = sample_config()
        experiment_config["config_version"] = CONFIG_VERSION
        with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
            "triage.util.conf.open", side_effect=open_side_effect
        ) as mock_file:
            db_engine = create_engine(postgresql.url())
            experiment = SingleThreadedExperiment(
                config=experiment_config,
                db_engine=db_engine,
                project_path=os.path.join(temp_dir, "inspections"),
            )

        assert isinstance(experiment, SingleThreadedExperiment)

    def test_noload_if_wrong_version(self):
        experiment_config = sample_config()
        experiment_config["config_version"] = "v0"
        with TemporaryDirectory() as temp_dir, mock.patch(
            "triage.util.conf.open", side_effect=open_side_effect
        ) as mock_file:
            with self.assertRaises(ValueError):
                SingleThreadedExperiment(
                    config=experiment_config,
                    db_engine=None,
                    project_path=os.path.join(temp_dir, "inspections"),
                )


@parametrize_experiment_classes
@mock.patch(
    "triage.component.architect.entity_date_table_generators."
    "EntityDateTableGenerator.clean_up",
    side_effect=lambda: time.sleep(1),
)
def test_cleanup_timeout(_clean_up_mock, experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
def test_build_error(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
    "triage.component.architect.entity_date_table_generators."
    "EntityDateTableGenerator.clean_up",
    side_effect=lambda: time.sleep(1),
)
def test_build_error_cleanup_timeout(_clean_up_mock, experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
def test_custom_label_name(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
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
    with TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        project_path = os.path.join(temp_dir, "inspections")
        SingleThreadedExperiment(
            config=sample_config(),
            db_engine=db_engine,
            project_path=project_path,
            profile=True,
        ).run()
        assert len(os.listdir(os.path.join(project_path, "profiling_stats"))) == 1


@parametrize_experiment_classes
def test_baselines_with_missing_features(experiment_class):
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = create_engine(postgresql.url())
        populate_source_data(db_engine)

        # set up the config with the baseline model and feature group mixing
        config = sample_config()
        config["grid_config"] = {
            "triage.component.catwalk.baselines.rankers.PercentileRankOneFeature": {
                "feature": ["entity_features_entity_id_1year_cat_sightings_count"]
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

        # assert
        # 1. that model groups entries are present
        num_mgs = len(
            [
                row
                for row in db_engine.execute(
                    "select * from triage_metadata.model_groups"
                )
            ]
        )
        assert num_mgs > 0

        # 2. that model entries are present, and linked to model groups
        num_models = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from triage_metadata.model_groups
                join triage_metadata.models using (model_group_id)
                where model_comment = 'test2-final-final'
            """
                )
            ]
        )
        assert num_models > 0

        # 3. predictions, linked to models
        num_predictions = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from test_results.predictions
                join triage_metadata.models using (model_id)"""
                )
            ]
        )
        assert num_predictions > 0

        # 4. evaluations linked to predictions linked to models
        num_evaluations = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from test_results.evaluations e
                join triage_metadata.models using (model_id)
                join test_results.predictions p on (
                    e.model_id = p.model_id and
                    e.evaluation_start_time <= p.as_of_date and
                    e.evaluation_end_time >= p.as_of_date)
            """
                )
            ]
        )
        assert num_evaluations > 0

        # 5. experiment
        num_experiments = len(
            [
                row
                for row in db_engine.execute(
                    "select * from triage_metadata.experiments"
                )
            ]
        )
        assert num_experiments == 1

        # 6. that models are linked to experiments
        num_models_with_experiment = len(
            [
                row
                for row in db_engine.execute(
                    """
                select * from triage_metadata.experiments
                join triage_metadata.experiment_models using (experiment_hash)
                join triage_metadata.models using (model_hash)
            """
                )
            ]
        )
        assert num_models == num_models_with_experiment

        # 7. that models have the train end date and label timespan
        results = [
            (model["train_end_time"], model["training_label_timespan"])
            for model in db_engine.execute("select * from triage_metadata.models")
        ]
        assert sorted(set(results)) == [
            (datetime(2012, 6, 1), timedelta(180)),
            (datetime(2013, 6, 1), timedelta(180)),
        ]

        # 8. that the right number of individual importances are present
        individual_importances = [
            row
            for row in db_engine.execute(
                """
            select * from test_results.individual_importances
            join triage_metadata.models using (model_id)
        """
            )
        ]
        assert len(individual_importances) == num_predictions * 2  # only 2 features


def test_serializable_engine_check_sqlalchemy_fail():
    """If we pass a vanilla sqlalchemy engine to the experiment we should blow up"""
    with testing.postgresql.Postgresql() as postgresql, TemporaryDirectory() as temp_dir, mock.patch(
        "triage.util.conf.open", side_effect=open_side_effect
    ) as mock_file:
        db_engine = sqlalchemy.create_engine(postgresql.url())
        with pytest.raises(TypeError):
            MultiCoreExperiment(
                config=sample_config(),
                db_engine=db_engine,
                project_path=os.path.join(temp_dir, "inspections"),
            )


def test_experiment_metadata(finished_experiment):
    session = sessionmaker(bind=finished_experiment.db_engine)()
    experiment_row = session.query(Experiment).get(finished_experiment.experiment_hash)
    assert experiment_row.time_splits == 2
    assert experiment_row.as_of_times == 369
    assert experiment_row.feature_blocks == 2
    assert experiment_row.feature_group_combinations == 1
    assert (
        experiment_row.matrices_needed
        == experiment_row.time_splits * 2 * experiment_row.feature_group_combinations
    )  # x2 for train and test
    assert experiment_row.grid_size == 4
    assert (
        experiment_row.models_needed
        == (experiment_row.matrices_needed / 2) * experiment_row.grid_size
    )  # /2 because we only need models per train matrix
    session.close()
