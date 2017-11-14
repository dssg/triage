import os
import time
from datetime import datetime, timedelta
from functools import partial
from tempfile import TemporaryDirectory
from unittest import mock, TestCase

import pytest
import testing.postgresql
from sqlalchemy import create_engine

from catwalk.db import ensure_db
from catwalk.storage import FSModelStorageEngine

from tests.utils import sample_config, populate_source_data

from triage.experiments import (
    MultiCoreExperiment,
    SingleThreadedExperiment,
    CONFIG_VERSION,
)


def num_linked_evaluations(db_engine):
    ((result,),) = db_engine.execute('''
        select count(*) from results.evaluations e
        join results.models using (model_id)
        join results.predictions p on (
            e.model_id = p.model_id and
            e.evaluation_start_time <= p.as_of_date and
            e.evaluation_end_time >= p.as_of_date)
    ''')
    return result


parametrize_experiment_classes = pytest.mark.parametrize(('experiment_class',), [
    (SingleThreadedExperiment,),
    (partial(MultiCoreExperiment, n_processes=2, n_db_processes=2),),
])


@parametrize_experiment_classes
def test_simple_experiment(experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections')
            ).run()

        # assert
        # 1. that model groups entries are present
        num_mgs = len([
            row for row in
            db_engine.execute('select * from results.model_groups')
        ])
        assert num_mgs > 0

        # 2. that model entries are present, and linked to model groups
        num_models = len([
            row for row in db_engine.execute('''
                select * from results.model_groups
                join results.models using (model_group_id)
                where model_comment = 'test2-final-final'
            ''')
        ])
        assert num_models > 0

        # 3. predictions, linked to models
        num_predictions = len([
            row for row in db_engine.execute('''
                select * from results.predictions
                join results.models using (model_id)''')
        ])
        assert num_predictions > 0

        # 4. evaluations linked to predictions linked to models
        num_evaluations = len([
            row for row in db_engine.execute('''
                select * from results.evaluations e
                join results.models using (model_id)
                join results.predictions p on (
                    e.model_id = p.model_id and
                    e.evaluation_start_time <= p.as_of_date and
                    e.evaluation_end_time >= p.as_of_date)
            ''')
        ])
        assert num_evaluations > 0

        # 5. experiment
        num_experiments = len([
            row for row in db_engine.execute('select * from results.experiments')
        ])
        assert num_experiments == 1

        # 6. that models are linked to experiments
        num_models_with_experiment = len([
            row for row in db_engine.execute('''
                select * from results.experiments
                join results.models using (experiment_hash)
            ''')
        ])
        assert num_models == num_models_with_experiment

        # 7. that models have the train end date and label timespan
        results = [
            (model['train_end_time'], model['training_label_timespan'])
            for model in db_engine.execute('select * from results.models')
        ]
        assert sorted(set(results)) == [
            (datetime(2012, 6, 1), timedelta(180)),
            (datetime(2013, 6, 1), timedelta(180)),
        ]

        # 8. that the right number of individual importances are present
        individual_importances = [row for row in db_engine.execute('''
            select * from results.individual_importances
            join results.models using (model_id)
        ''')]
        assert len(individual_importances) == num_predictions * 2  # only 2 features


@parametrize_experiment_classes
def test_restart_experiment(experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
            )
            experiment.run()

            evaluations = num_linked_evaluations(db_engine)
            assert evaluations > 0

            experiment = experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                replace=False
            )
            experiment.make_entity_date_table = mock.Mock()
            experiment.run()
            assert not experiment.make_entity_date_table.called


@parametrize_experiment_classes
def test_nostate_experiment(experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        experiment_config = sample_config()
        del experiment_config['state_config']
        with TemporaryDirectory() as temp_dir:
            exp = experiment_class(
                config=experiment_config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections')
            )
            exp.run()

        # assert
        # 1. that model groups entries are present
        num_mgs = len([
            row for row in
            db_engine.execute('select * from results.model_groups')
        ])
        assert num_mgs > 0

        # 2. that model entries are present, and linked to model groups
        num_models = len([
            row for row in db_engine.execute('''
                select * from results.model_groups
                join results.models using (model_group_id)
                where model_comment = 'test2-final-final'
            ''')
        ])
        assert num_models > 0

        # 3. predictions, linked to models
        num_predictions = len([
            row for row in db_engine.execute('''
                select * from results.predictions
                join results.models using (model_id)''')
        ])
        assert num_predictions > 0

        # 4. evaluations linked to predictions linked to models
        num_evaluations = len([
            row for row in db_engine.execute('''
                select * from results.evaluations e
                join results.models using (model_id)
                join results.predictions p on (
                    e.model_id = p.model_id and
                    e.evaluation_start_time <= p.as_of_date and
                    e.evaluation_end_time >= p.as_of_date)
            ''')
        ])
        assert num_evaluations > 0

        # 5. experiment
        num_experiments = len([
            row for row in db_engine.execute('select * from results.experiments')
        ])
        assert num_experiments == 1

        # 6. that models are linked to experiments
        num_models_with_experiment = len([
            row for row in db_engine.execute('''
                select * from results.experiments
                join results.models using (experiment_hash)
            ''')
        ])
        assert num_models == num_models_with_experiment

        # 7. that models have the train end date and label timespan
        results = [
            (model['train_end_time'], model['training_label_timespan'])
            for model in db_engine.execute('select * from results.models')
        ]
        assert sorted(set(results)) == [
            (datetime(2012, 6, 1), timedelta(180)),
            (datetime(2013, 6, 1), timedelta(180)),
        ]


class TestConfigVersion(TestCase):

    def test_load_if_right_version(self):
        experiment_config = sample_config()
        experiment_config['config_version'] = CONFIG_VERSION
        with testing.postgresql.Postgresql() as postgresql:
            db_engine = create_engine(postgresql.url())
            ensure_db(db_engine)
            with TemporaryDirectory() as temp_dir:
                experiment = SingleThreadedExperiment(
                    config=experiment_config,
                    db_engine=db_engine,
                    model_storage_class=FSModelStorageEngine,
                    project_path=os.path.join(temp_dir, 'inspections'),
                )

        assert isinstance(experiment, SingleThreadedExperiment)

    def test_noload_if_wrong_version(self):
        experiment_config = sample_config()
        experiment_config['config_version'] = 'v0'
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                SingleThreadedExperiment(
                    config=experiment_config,
                    db_engine=None,
                    model_storage_class=FSModelStorageEngine,
                    project_path=os.path.join(temp_dir, 'inspections'),
                )


@parametrize_experiment_classes
@mock.patch('architect.state_table_generators.StateTableGenerator.clean_up',
            side_effect=lambda: time.sleep(1))
def test_cleanup_timeout(_clean_up_mock, experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup_timeout=0.02,  # Set short timeout
            )
            with pytest.raises(TimeoutError):
                experiment()


@parametrize_experiment_classes
def test_build_error(experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        with TemporaryDirectory() as temp_dir:
            experiment = experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
            )

            with mock.patch.object(experiment, 'build_matrices') as build_mock:
                build_mock.side_effect = RuntimeError('boom!')

                with pytest.raises(RuntimeError):
                    experiment()


@parametrize_experiment_classes
@mock.patch('architect.state_table_generators.StateTableGenerator.clean_up',
            side_effect=lambda: time.sleep(1))
def test_build_error_cleanup_timeout(_clean_up_mock, experiment_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        with TemporaryDirectory() as temp_dir:
            experiment = experiment_class(
                config=sample_config(),
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup_timeout=0.02,  # Set short timeout
            )

            with mock.patch.object(experiment, 'build_matrices') as build_mock:
                build_mock.side_effect = RuntimeError('boom!')

                with pytest.raises(TimeoutError) as exc_info:
                    experiment()

    # Last exception is TimeoutError, but earlier error is preserved in
    # __context__, and will be noted as well in any standard traceback:
    assert exc_info.value.__context__ is build_mock.side_effect
