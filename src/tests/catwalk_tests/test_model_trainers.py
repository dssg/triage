import pandas
import testing.postgresql
import sqlalchemy
import unittest
from unittest.mock import patch
import pytest

from sqlalchemy import create_engine
from triage.component.catwalk.db import ensure_db
from tests.results_tests.factories import init_engine

from triage.component.catwalk.model_grouping import ModelGrouper
from triage.component.catwalk.model_trainers import ModelTrainer
from tests.utils import rig_engines, get_matrix_store


@pytest.fixture
def grid_config():
    return {
        "sklearn.tree.DecisionTreeClassifier": {
            "min_samples_split": [10, 100],
            "max_depth": [3, 5],
            "criterion": ["gini"],
        }
    }


def test_model_trainer(grid_config):
    with rig_engines() as (db_engine, project_storage):
        # Creates a matrix entry in the matrices table with uuid from metadata above
        model_storage_engine = project_storage.model_storage_engine()
        trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=model_storage_engine,
            model_grouper=ModelGrouper(),
            db_engine=db_engine,
        )
        model_ids = trainer.train_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )

        # assert
        # 1. that the models and feature importances table entries are present
        records = [
            row
            for row in db_engine.execute(
                "select * from train_results.feature_importances"
            )
        ]
        assert len(records) == 4 * 2  # maybe exclude entity_id? yes

        records = [
            row
            for row in db_engine.execute("select model_hash from model_metadata.models")
        ]
        assert len(records) == 4
        hashes = [row[0] for row in records]

        # 2. that the model groups are distinct
        records = [
            row
            for row in db_engine.execute(
                "select distinct model_group_id from model_metadata.models"
            )
        ]
        assert len(records) == 4

        # 3. that the model sizes are saved in the table and all are < 1 kB
        records = [
            row
            for row in db_engine.execute("select model_size from model_metadata.models")
        ]
        assert len(records) == 4
        for i in records:
            size = i[0]
            assert size < 1

        # 4. that all four models are cached
        model_pickles = [model_storage_engine.load(model_hash) for model_hash in hashes]
        assert len(model_pickles) == 4
        assert len([x for x in model_pickles if x is not None]) == 4

        # 5. that their results can have predictions made on it
        test_matrix = pandas.DataFrame.from_dict(
            {"entity_id": [3, 4], "feature_one": [4, 4], "feature_two": [6, 5]}
        ).set_index("entity_id")

        for model_pickle in model_pickles:
            predictions = model_pickle.predict(test_matrix)
            assert len(predictions) == 2

        # 6. when run again, same models are returned
        new_model_ids = trainer.train_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )
        assert (
            len(
                [
                    row
                    for row in db_engine.execute(
                        "select model_hash from model_metadata.models"
                    )
                ]
            )
            == 4
        )
        assert model_ids == new_model_ids

        # 7. if replace is set, update non-unique attributes and feature importances
        max_batch_run_time = [
            row[0]
            for row in db_engine.execute(
                "select max(batch_run_time) from model_metadata.models"
            )
        ][0]
        trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=model_storage_engine,
            model_grouper=ModelGrouper(
                model_group_keys=["label_name", "label_timespan"]
            ),
            db_engine=db_engine,
            replace=True,
        )
        new_model_ids = trainer.train_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )
        assert model_ids == new_model_ids
        assert [
            row["model_id"]
            for row in db_engine.execute(
                "select model_id from model_metadata.models order by 1 asc"
            )
        ] == model_ids
        new_max_batch_run_time = [
            row[0]
            for row in db_engine.execute(
                "select max(batch_run_time) from model_metadata.models"
            )
        ][0]
        assert new_max_batch_run_time > max_batch_run_time

        records = [
            row
            for row in db_engine.execute(
                "select * from train_results.feature_importances"
            )
        ]
        assert len(records) == 4 * 2  # maybe exclude entity_id? yes

        # 8. if the cache is missing but the metadata is still there, reuse the metadata
        for row in db_engine.execute("select model_hash from model_metadata.models"):
            model_storage_engine.delete(row[0])
        new_model_ids = trainer.train_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )
        assert model_ids == sorted(new_model_ids)

        # 9. that the generator interface works the same way
        new_model_ids = trainer.generate_trained_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )
        assert model_ids == sorted([model_id for model_id in new_model_ids])


def test_baseline_exception_handling():
    grid_config = {
        "triage.component.catwalk.baselines.rankers.PercentileRankOneFeature": {
            "feature": ["feature_one", "feature_three"]
        }
    }
    with rig_engines() as (db_engine, project_storage):
        trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=project_storage.model_storage_engine(),
            db_engine=db_engine,
            model_grouper=ModelGrouper(),
        )

        train_tasks = trainer.generate_train_tasks(
            grid_config, dict(), get_matrix_store(project_storage)
        )

        model_ids = []
        for train_task in train_tasks:
            model_ids.append(trainer.process_train_task(**train_task))
        assert model_ids == [1, None]


def test_custom_groups(grid_config):
    with rig_engines() as (db_engine, project_storage):
        # create training set
        model_storage_engine = project_storage.model_storage_engine()
        trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=model_storage_engine,
            model_grouper=ModelGrouper(["class_path"]),
            db_engine=db_engine,
        )
        model_ids = trainer.train_models(
            grid_config=grid_config,
            misc_db_parameters=dict(),
            matrix_store=get_matrix_store(project_storage),
        )
        # expect only one model group now
        records = [
            row[0]
            for row in db_engine.execute(
                "select distinct model_group_id from model_metadata.models"
            )
        ]
        assert len(records) == 1
        assert records[0] == model_ids[0]


def test_n_jobs_not_new_model():
    grid_config = {
        "sklearn.ensemble.AdaBoostClassifier": {"n_estimators": [10, 100, 1000]},
        "sklearn.ensemble.RandomForestClassifier": {
            "n_estimators": [10, 100],
            "max_features": ["sqrt", "log2"],
            "max_depth": [5, 10, 15, 20],
            "criterion": ["gini", "entropy"],
            "n_jobs": [12, 24],
        },
    }

    with rig_engines() as (db_engine, project_storage):
        model_storage_engine = project_storage.model_storage_engine()
        trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=model_storage_engine,
            db_engine=db_engine,
            model_grouper=ModelGrouper(),
        )

        train_tasks = trainer.generate_train_tasks(
            grid_config, dict(), get_matrix_store(project_storage)
        )

        assert len(train_tasks) == 35  # 32+3, would be (32*2)+3 if we didn't remove
        assert (
            len([task for task in train_tasks if "n_jobs" in task["parameters"]]) == 32
        )

        for train_task in train_tasks:
            trainer.process_train_task(**train_task)

        for row in db_engine.execute(
            "select hyperparameters from model_metadata.model_groups"
        ):
            assert "n_jobs" not in row[0]


class RetryTest(unittest.TestCase):
    def test_retry_max(self):
        db_engine = None
        trainer = None
        # set up a basic model training run
        with rig_engines() as (db_engine, project_storage):
            trainer = ModelTrainer(
                experiment_hash=None,
                model_storage_engine=project_storage.model_storage_engine(),
                db_engine=db_engine,
                model_grouper=ModelGrouper(),
            )
            matrix_store = get_matrix_store(project_storage)

        # the postgres server goes out of scope here and thus no longer exists
        with patch("time.sleep") as time_mock:
            with self.assertRaises(sqlalchemy.exc.OperationalError):
                trainer.train_models(grid_config(), dict(), matrix_store)
            # we want to make sure that we are using the retrying module sanely
            # as opposed to matching the exact # of calls specified by the code
            assert len(time_mock.mock_calls) > 5

    def test_retry_recovery(self):
        db_engine = None
        trainer = None
        port = None
        with rig_engines() as (db_engine, project_storage):
            port = db_engine.url.port
            trainer = ModelTrainer(
                experiment_hash=None,
                model_storage_engine=project_storage.model_storage_engine(),
                db_engine=db_engine,
                model_grouper=ModelGrouper(),
            )
            matrix_store = get_matrix_store(project_storage)

        # start without a database server
        # then bring it back up after the first sleep
        # use self so it doesn't go out of scope too early and shut down
        self.new_server = None

        def replace_db(arg):
            self.new_server = testing.postgresql.Postgresql(port=port)
            db_engine = create_engine(self.new_server.url())
            ensure_db(db_engine)
            init_engine(db_engine)
            get_matrix_store(project_storage)

        with patch("time.sleep") as time_mock:
            time_mock.side_effect = replace_db
            try:
                trainer.train_models(grid_config(), dict(), matrix_store)
            finally:
                if self.new_server is not None:
                    self.new_server.stop()
            assert len(time_mock.mock_calls) == 1
