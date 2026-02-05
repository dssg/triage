# from sqlalchemy.orm import Session
import datetime
from unittest import mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.results_tests.factories import (
    ExperimentFactory,
    TriageRunFactory,
    clear_session,
    set_session,
)
from tests.utils import open_side_effect, populate_source_data, sample_config
from triage.component.results_schema import TriageRun, TriageRunStatus
from triage.experiments import MultiCoreExperiment, SingleThreadedExperiment
from triage.tracking import (
    get_run_for_update,
    increment_field,
    initialize_tracking_and_get_run_id,
)
from triage.util.db import scoped_session


@pytest.fixture(name="test_engine", scope="function")
def shared_db_engine_with_source_data(shared_db_engine):
    """A successfully-run experiment. Its database schemas and project storage can be queried.

    Returns: (triage.experiments.SingleThreadedExperiment)
    """
    populate_source_data(shared_db_engine)
    yield shared_db_engine


@pytest.mark.skip(
    reason="Pre-existing issue: pip_freeze returns empty list in test environment"
)
def test_experiment_tracker(test_engine, project_path):
    SessionLocal = sessionmaker(bind=test_engine, future=True)
    session = SessionLocal()

    with mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file:
        experiment = MultiCoreExperiment(
            config=sample_config(),
            db_engine=test_engine,
            project_path=project_path,
            n_processes=4,
        )

    try:
        set_session(session)
        # with Session(test_engine) as session:
        experiment_run = session.get(TriageRun, experiment.run_id)
        assert experiment_run.current_status == TriageRunStatus.started
        assert experiment_run.run_hash == experiment.experiment_hash
        assert experiment_run.run_type == "experiment"
        assert (
            experiment_run.experiment_class_path
            == "triage.experiments.multicore.MultiCoreExperiment"
        )
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
    finally:
        clear_session()
        session.close()

    try:
        set_session(session)
        # with Session(test_engine) as session:
        experiment_run = session.get(TriageRun, experiment.run_id)
        assert experiment_run.start_method == "run"
        assert experiment_run.matrices_made == len(experiment.matrix_build_tasks)
        assert experiment_run.matrices_skipped == 0
        assert experiment_run.matrices_errored == 0
        assert experiment_run.models_skipped == 0
        assert experiment_run.models_errored == 0
        assert experiment_run.models_made == len(
            list(
                task["train_kwargs"]["model_hash"]
                for batch in experiment._all_train_test_batches()
                for task in batch.tasks
            )
        )
        assert isinstance(experiment_run.matrix_building_started, datetime.datetime)
        assert isinstance(experiment_run.model_building_started, datetime.datetime)
        assert isinstance(experiment_run.last_updated_time, datetime.datetime)
        assert not experiment_run.stacktrace
        assert experiment_run.current_status == TriageRunStatus.completed
    finally:
        clear_session()
        session.close()


def test_experiment_tracker_exception(db_engine, project_path):
    SessionLocal = sessionmaker(bind=db_engine, future=True)
    session = SessionLocal()

    try:
        set_session(session)
        with mock.patch(
            "triage.util.conf.open", side_effect=open_side_effect
        ) as mock_file:
            experiment = SingleThreadedExperiment(
                config=sample_config(),
                db_engine=db_engine,
                project_path=project_path,
            )
        # no source data means this should blow up
        with pytest.raises(Exception):
            experiment.run()

        experiment_run = session.get(TriageRun, experiment.run_id)
        assert experiment_run.current_status == TriageRunStatus.failed
        assert isinstance(experiment_run.last_updated_time, datetime.datetime)
        assert experiment_run.stacktrace
    finally:
        clear_session()
        session.close()


def test_experiment_tracker_in_parts(test_engine, project_path):
    SessionLocal = sessionmaker(bind=test_engine, future=True)
    session = SessionLocal()

    with mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file:
        experiment = SingleThreadedExperiment(
            config=sample_config(),
            db_engine=test_engine,
            project_path=project_path,
        )

    try:
        set_session(session)

        experiment.generate_matrices()

        with scoped_session(test_engine) as session:
            experiment_run = session.get(TriageRun, experiment.run_id)
            assert experiment_run.start_method == "generate_matrices"
    finally:
        clear_session()
        session.close()


def test_initialize_tracking_and_get_run_id(db_engine_with_results_schema):
    SessionLocal = sessionmaker(bind=db_engine_with_results_schema, future=True)
    session = SessionLocal()

    try:
        set_session(session)

        experiment = ExperimentFactory()
        session.commit()
        experiment_hash = experiment.experiment_hash
        run_id = initialize_tracking_and_get_run_id(
            experiment_hash=experiment_hash,
            experiment_class_path="mymodule.MyClassName",
            random_seed=1234,
            experiment_kwargs={"key": "value"},
            db_engine=db_engine_with_results_schema,
        )
        assert run_id

        # with scoped_session(db_engine_with_results_schema) as session:
        experiment_run = session.get(TriageRun, run_id)
        assert experiment_run.run_hash == experiment_hash
        assert experiment_run.experiment_class_path == "mymodule.MyClassName"
        assert experiment_run.random_seed == 1234
        assert experiment_run.experiment_kwargs == {"key": "value"}
        new_run_id = initialize_tracking_and_get_run_id(
            experiment_hash=experiment_hash,
            experiment_class_path="mymodule.MyClassName",
            random_seed=5432,
            experiment_kwargs={"key": "value"},
            db_engine=db_engine_with_results_schema,
        )
        assert new_run_id > run_id
    finally:
        clear_session()
        session.close()


def test_get_run_for_update(db_engine_with_results_schema):
    SessionLocal = sessionmaker(bind=db_engine_with_results_schema, future=True)
    session = SessionLocal()

    try:
        set_session(session)

        experiment_run = TriageRunFactory()
        session.commit()
        with get_run_for_update(
            db_engine=db_engine_with_results_schema, run_id=experiment_run.run_id
        ) as run_obj:
            run_obj.stacktrace = "My stacktrace"

        experiment_run_from_db = session.get(TriageRun, experiment_run.run_id)
        assert experiment_run_from_db.stacktrace == "My stacktrace"
    finally:
        clear_session()
        session.close()


def test_increment_field(db_engine_with_results_schema):
    SessionLocal = sessionmaker(bind=db_engine_with_results_schema, future=True)
    session = SessionLocal()

    try:
        set_session(session)

        experiment_run = TriageRunFactory()
        session.commit()
        increment_field(
            "matrices_made", experiment_run.run_id, db_engine_with_results_schema
        )
        increment_field(
            "matrices_made", experiment_run.run_id, db_engine_with_results_schema
        )

        experiment_run_from_db = session.get(TriageRun, experiment_run.run_id)
        assert experiment_run_from_db.matrices_made == 2
    finally:
        clear_session()
        session.close()
