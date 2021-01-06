import pandas as pd
import random
import pytest


from triage.component.catwalk.model_grouping import ModelGrouper
from triage.component.catwalk.model_trainers import ModelTrainer
from tests.utils import get_matrix_store


@pytest.fixture
def grid_config():
    return {
        "sklearn.tree.DecisionTreeClassifier": {
            "min_samples_split": [10, 100],
            "max_depth": [3, 5],
            "criterion": ["gini"],
        }
    }


@pytest.fixture(scope="function")
def default_model_trainer(db_engine_with_results_schema, project_storage):
    model_storage_engine = project_storage.model_storage_engine()
    trainer = ModelTrainer(
        experiment_hash=None,
        model_storage_engine=model_storage_engine,
        db_engine=db_engine_with_results_schema,
        model_grouper=ModelGrouper(),
    )
    yield trainer


def test_model_trainer(grid_config, default_model_trainer):
    trainer = default_model_trainer
    db_engine = trainer.db_engine
    project_storage = trainer.model_storage_engine.project_storage
    model_storage_engine = trainer.model_storage_engine

    def set_test_seed():
        random.seed(5)
    set_test_seed()
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
        for row in db_engine.execute("select model_hash from triage_metadata.models")
    ]
    assert len(records) == 4
    hashes = [row[0] for row in records]

    # 2. that the model groups are distinct
    records = [
        row
        for row in db_engine.execute(
            "select distinct model_group_id from triage_metadata.models"
        )
    ]
    assert len(records) == 4

    # 2. that the random seeds are distinct
    records = [
        row
        for row in db_engine.execute(
            "select distinct random_seed from triage_metadata.models"
        )
    ]
    assert len(records) == 4

    # 3. that the model sizes are saved in the table and all are < 1 kB
    records = [
        row
        for row in db_engine.execute("select model_size from triage_metadata.models")
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
    test_matrix = pd.DataFrame.from_dict(
        {"entity_id": [3, 4], "feature_one": [4, 4], "feature_two": [6, 5]}
    ).set_index("entity_id")

    for model_pickle in model_pickles:
        predictions = model_pickle.predict(test_matrix)
        assert len(predictions) == 2

    # 6. when run again with the same starting seed, same models are returned
    set_test_seed()
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
                    "select model_hash from triage_metadata.models"
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
            "select max(batch_run_time) from triage_metadata.models"
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
    set_test_seed()
    new_model_ids = trainer.train_models(
        grid_config=grid_config,
        misc_db_parameters=dict(),
        matrix_store=get_matrix_store(project_storage),
    )
    assert model_ids == new_model_ids
    assert [
        row["model_id"]
        for row in db_engine.execute(
            "select model_id from triage_metadata.models order by 1 asc"
        )
    ] == model_ids
    new_max_batch_run_time = [
        row[0]
        for row in db_engine.execute(
            "select max(batch_run_time) from triage_metadata.models"
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
    set_test_seed()
    for row in db_engine.execute("select model_hash from triage_metadata.models"):
        model_storage_engine.delete(row[0])
    new_model_ids = trainer.train_models(
        grid_config=grid_config,
        misc_db_parameters=dict(),
        matrix_store=get_matrix_store(project_storage),
    )
    assert model_ids == sorted(new_model_ids)

    # 9. that the generator interface works the same way
    set_test_seed()
    new_model_ids = trainer.generate_trained_models(
        grid_config=grid_config,
        misc_db_parameters=dict(),
        matrix_store=get_matrix_store(project_storage),
    )
    assert model_ids == sorted([model_id for model_id in new_model_ids])


def test_baseline_exception_handling(default_model_trainer):
    grid_config = {
        "triage.component.catwalk.baselines.rankers.PercentileRankOneFeature": {
            "feature": ["feature_one", "feature_three"]
        }
    }
    trainer = default_model_trainer
    project_storage = trainer.model_storage_engine.project_storage

    train_tasks = trainer.generate_train_tasks(
        grid_config, dict(), get_matrix_store(project_storage)
    )

    model_ids = []
    for train_task in train_tasks:
        model_ids.append(trainer.process_train_task(**train_task))
    assert model_ids == [1, None]


def test_custom_groups(grid_config, db_engine_with_results_schema, project_storage):
    model_storage_engine = project_storage.model_storage_engine()
    trainer = ModelTrainer(
        experiment_hash=None,
        model_storage_engine=model_storage_engine,
        model_grouper=ModelGrouper(["class_path"]),
        db_engine=db_engine_with_results_schema,
    )
    # create training set
    model_ids = trainer.train_models(
        grid_config=grid_config,
        misc_db_parameters=dict(),
        matrix_store=get_matrix_store(project_storage),
    )
    # expect only one model group now
    records = [
        row[0]
        for row in db_engine_with_results_schema.execute(
            "select distinct model_group_id from triage_metadata.models"
        )
    ]
    assert len(records) == 1
    assert records[0] == model_ids[0]


def test_n_jobs_not_new_model(default_model_trainer):
    grid_config = {
        "sklearn.ensemble.AdaBoostClassifier": {"n_estimators": [10, 100, 1000]},
        "sklearn.ensemble.RandomForestClassifier": {
            "n_estimators": [10, 100],
            "max_features": ["sqrt", "log2"],
            "max_depth": [5, 10, 15, 20],
            "criterion": ["gini", "entropy"],
            "n_jobs": [12],
        },
    }

    trainer = default_model_trainer
    project_storage = trainer.model_storage_engine.project_storage
    db_engine = trainer.db_engine

    # generate train tasks, with a specific random seed so that we can compare
    # apples to apples later
    random.seed(5)
    train_tasks = trainer.generate_train_tasks(
        grid_config, dict(), get_matrix_store(project_storage)
    )

    for train_task in train_tasks:
        trainer.process_train_task(**train_task)

    # since n_jobs is a runtime attribute of the model, it should not make it
    # into the model group
    for row in db_engine.execute(
        "select hyperparameters from triage_metadata.model_groups"
    ):
        assert "n_jobs" not in row[0]

    hashes = set(task['model_hash'] for task in train_tasks)
    # generate the grid again with a different n_jobs (but the same random seed!)
    # and make sure that the hashes are the same as before
    random.seed(5)
    grid_config['sklearn.ensemble.RandomForestClassifier']['n_jobs'] = [24]
    new_train_tasks = trainer.generate_train_tasks(
        grid_config, dict(), get_matrix_store(project_storage)
    )
    assert hashes == set(task['model_hash'] for task in new_train_tasks)


def test_cache_models(default_model_trainer):
    assert not default_model_trainer.model_storage_engine.should_cache
    with default_model_trainer.cache_models():
        assert default_model_trainer.model_storage_engine.should_cache
    assert not default_model_trainer.model_storage_engine.should_cache
