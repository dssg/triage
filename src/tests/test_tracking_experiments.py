from tests.utils import sample_config, populate_source_data
from triage.util.db import scoped_session
from triage.experiments import MultiCoreExperiment, SingleThreadedExperiment
from triage.component.results_schema import ExperimentRun, ExperimentRunStatus
from tests.results_tests.factories import ExperimentFactory, ExperimentRunFactory, session as factory_session
from sqlalchemy.orm import Session
import pytest
import datetime
from triage.tracking import (
    initialize_tracking_and_get_run_id,
    get_run_for_update,
    increment_field
)


@pytest.fixture(name="test_engine", scope="module")
def shared_db_engine_with_source_data(shared_db_engine):
    """A successfully-run experiment. Its database schemas and project storage can be queried.

    Returns: (triage.experiments.SingleThreadedExperiment)
    """
    populate_source_data(shared_db_engine)
    yield shared_db_engine


def test_experiment_tracker(test_engine, project_path):
    experiment = MultiCoreExperiment(
        config=sample_config(),
        db_engine=test_engine,
        project_path=project_path,
        n_processes=4,
    )
    experiment_run = Session(bind=test_engine).query(ExperimentRun).get(experiment.run_id)
    assert experiment_run.current_status == ExperimentRunStatus.started
    assert experiment_run.experiment_hash == experiment.experiment_hash
    assert experiment_run.experiment_class_path == 'triage.experiments.multicore.MultiCoreExperiment'
    assert experiment_run.platform
    assert experiment_run.os_user
    assert experiment_run.installed_libraries
    assert experiment_run.matrices_skipped == 0
    assert experiment_run.matrices_errored == 0
    assert experiment_run.matrices_made == 0
    assert experiment_run.models_skipped == 0
    assert experiment_run.models_errored == 0
    assert experiment_run.models_made == 0

    experiment.run()
    experiment_run = Session(bind=test_engine).query(ExperimentRun).get(experiment.run_id)
    assert experiment_run.start_method == "run"
    assert experiment_run.matrices_made == len(experiment.matrix_build_tasks)
    assert experiment_run.matrices_skipped == 0
    assert experiment_run.matrices_errored == 0
    assert experiment_run.models_skipped == 0
    assert experiment_run.models_errored == 0
    assert experiment_run.models_made == len(list(task['train_kwargs']['model_hash'] for batch in experiment._all_train_test_batches() for task in batch.tasks))
    assert isinstance(experiment_run.matrix_building_started, datetime.datetime)
    assert isinstance(experiment_run.model_building_started, datetime.datetime)
    assert isinstance(experiment_run.last_updated_time, datetime.datetime)
    assert not experiment_run.stacktrace
    assert experiment_run.current_status == ExperimentRunStatus.completed


def test_experiment_tracker_exception(db_engine, project_path):
    experiment = SingleThreadedExperiment(
        config=sample_config(),
        db_engine=db_engine,
        project_path=project_path,
    )
    # no source data means this should blow up
    with pytest.raises(Exception):
        experiment.run()

    with scoped_session(db_engine) as session:
        experiment_run = session.query(ExperimentRun).get(experiment.run_id)
        assert experiment_run.current_status == ExperimentRunStatus.failed
        assert isinstance(experiment_run.last_updated_time, datetime.datetime)
        assert experiment_run.stacktrace


def test_experiment_tracker_in_parts(test_engine, project_path):
    experiment = SingleThreadedExperiment(
        config=sample_config(),
        db_engine=test_engine,
        project_path=project_path,
    )
    experiment.generate_matrices()
    experiment.train_and_test_models()
    with scoped_session(test_engine) as session:
        experiment_run = session.query(ExperimentRun).get(experiment.run_id)
        assert experiment_run.start_method == "generate_matrices"


def test_initialize_tracking_and_get_run_id(db_engine_with_results_schema):
    experiment = ExperimentFactory()
    factory_session.commit()
    experiment_hash = experiment.experiment_hash
    run_id = initialize_tracking_and_get_run_id(
        experiment_hash=experiment_hash,
        experiment_class_path='mymodule.MyClassName',
        random_seed=1234,
        experiment_kwargs={'key': 'value'},
        db_engine=db_engine_with_results_schema
    )
    assert run_id
    with scoped_session(db_engine_with_results_schema) as session:
        experiment_run = session.query(ExperimentRun).get(run_id)
        assert experiment_run.experiment_hash == experiment_hash
        assert experiment_run.experiment_class_path == 'mymodule.MyClassName'
        assert experiment_run.random_seed == 1234
        assert experiment_run.experiment_kwargs == {'key': 'value'}
    new_run_id = initialize_tracking_and_get_run_id(
        experiment_hash=experiment_hash,
        experiment_class_path='mymodule.MyClassName',
        random_seed=5432,
        experiment_kwargs={'key': 'value'},
        db_engine=db_engine_with_results_schema
    )
    assert new_run_id > run_id


def test_get_run_for_update(db_engine_with_results_schema):
    experiment_run = ExperimentRunFactory()
    factory_session.commit()
    with get_run_for_update(
        db_engine=db_engine_with_results_schema,
        run_id=experiment_run.run_id
    ) as run_obj:
        run_obj.stacktrace = "My stacktrace"

    with scoped_session(db_engine_with_results_schema) as session:
        experiment_run_from_db = session.query(ExperimentRun).get(experiment_run.run_id)
        assert experiment_run_from_db.stacktrace == "My stacktrace"


def test_increment_field(db_engine_with_results_schema):
    experiment_run = ExperimentRunFactory()
    factory_session.commit()
    increment_field('matrices_made', experiment_run.run_id, db_engine_with_results_schema)
    increment_field('matrices_made', experiment_run.run_id, db_engine_with_results_schema)

    with scoped_session(db_engine_with_results_schema) as session:
        experiment_run_from_db = session.query(ExperimentRun).get(experiment_run.run_id)
        assert experiment_run_from_db.matrices_made == 2
