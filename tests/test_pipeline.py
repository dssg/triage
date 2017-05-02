from datetime import datetime
import os
from sqlalchemy import create_engine
from functools import partial
from tempfile import TemporaryDirectory
import testing.postgresql
from unittest.mock import Mock

from triage.db import ensure_db
from triage.storage import FSModelStorageEngine

from triage.pipelines import LocalParallelPipeline, SerialPipeline


def populate_source_data(db_engine):
    complaints = [
        (1, '2010-10-01', 5),
        (1, '2011-10-01', 4),
        (1, '2011-11-01', 4),
        (1, '2011-12-01', 4),
        (1, '2012-02-01', 5),
        (1, '2012-10-01', 4),
        (1, '2013-10-01', 5),
        (2, '2010-10-01', 5),
        (2, '2011-10-01', 5),
        (2, '2011-11-01', 4),
        (2, '2011-12-01', 4),
        (2, '2012-02-01', 6),
        (2, '2012-10-01', 5),
        (2, '2013-10-01', 6),
        (3, '2010-10-01', 5),
        (3, '2011-10-01', 3),
        (3, '2011-11-01', 4),
        (3, '2011-12-01', 4),
        (3, '2012-02-01', 4),
        (3, '2012-10-01', 3),
        (3, '2013-10-01', 4),
    ]

    events = [
        (1, 1, '2011-01-01'),
        (1, 1, '2011-06-01'),
        (1, 1, '2011-09-01'),
        (1, 1, '2012-01-01'),
        (1, 1, '2012-01-10'),
        (1, 1, '2012-06-01'),
        (1, 1, '2013-01-01'),
        (1, 0, '2014-01-01'),
        (1, 1, '2015-01-01'),
        (2, 1, '2011-01-01'),
        (2, 1, '2011-06-01'),
        (2, 1, '2011-09-01'),
        (2, 1, '2012-01-01'),
        (2, 1, '2013-01-01'),
        (2, 1, '2014-01-01'),
        (2, 1, '2015-01-01'),
        (3, 0, '2011-01-01'),
        (3, 0, '2011-06-01'),
        (3, 0, '2011-09-01'),
        (3, 0, '2012-01-01'),
        (3, 0, '2013-01-01'),
        (3, 1, '2014-01-01'),
        (3, 0, '2015-01-01'),
    ]

    db_engine.execute('''create table cat_complaints (
        entity_id int,
        as_of_date date,
        cat_sightings int
        )''')

    for complaint in complaints:
        db_engine.execute(
            "insert into cat_complaints values (%s, %s, %s)",
            complaint
        )

    db_engine.execute('''create table events (
        entity_id int,
        outcome int,
        outcome_date date
    )''')

    for event in events:
        db_engine.execute(
            "insert into events values (%s, %s, %s)",
            event
        )

def num_linked_evaluations(db_engine):
    num_evaluations = len([
        row for row in db_engine.execute('''
            select * from results.evaluations e
            join results.models using (model_id)
            join results.predictions p on (
                e.model_id = p.model_id and
                e.evaluation_start_time <= p.as_of_date and
                e.evaluation_end_time > p.as_of_date)
        ''')
    ])
    return num_evaluations


def simple_pipeline_test(pipeline_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        temporal_config = {
            'beginning_of_time': '2010-01-01',
            'modeling_start_time': '2011-01-01',
            'modeling_end_time': '2014-01-01',
            'update_window': '1y',
            'prediction_window': '6months',
            'look_back_durations': ['6months'],
            'train_example_frequency': '1day',
            'test_example_frequency': '3months',
            'test_durations': ['1months'],
            'prediction_frequency': '1d'
        }
        scoring_config = {
            'metric_groups': [
                {'metrics': ['precision@'], 'thresholds': {'top_n': [2]}}
            ]
        }
        grid_config = {
            'sklearn.linear_model.LogisticRegression': {
                'C': [0.00001, 0.0001],
                'penalty': ['l1', 'l2'],
                'random_state': [2193]
            }
        }
        feature_config = [{
            'prefix': 'test_features',
            'from_obj': 'cat_complaints',
            'knowledge_date_column': 'as_of_date',
            'aggregates': [{
                'quantity': 'cat_sightings',
                'metrics': ['count', 'avg'],
            }],
            'intervals': ['1y'],
            'groups': ['entity_id']
        }]
        experiment_config = {
            'events_table': 'events',
            'entity_column_name': 'entity_id',
            'model_comment': 'test2-final-final',
            'feature_aggregations': feature_config,
            'temporal_config': temporal_config,
            'grid_config': grid_config,
            'scoring': scoring_config,
        }

        with TemporaryDirectory() as temp_dir:
            pipeline_class(
                config=experiment_config,
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
                    e.evaluation_end_time > p.as_of_date)
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

        # 7. that models have the train end date, including prediction window
        train_end_times = [
            model['train_end_time']
            for model in db_engine.execute('select * from results.models')
        ]
        assert sorted(set(train_end_times)) == [datetime(2012, 7, 1), datetime(2013, 7, 1)]


def test_serial_pipeline():
    simple_pipeline_test(SerialPipeline)


def test_local_parallel_pipeline():
    simple_pipeline_test(
        partial(LocalParallelPipeline, n_processes=2, n_db_processes=2)
    )


def reuse_pipeline_test(pipeline_class):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        temporal_config = {
            'beginning_of_time': '2010-01-01',
            'modeling_start_time': '2011-01-01',
            'modeling_end_time': '2014-01-01',
            'update_window': '1y',
            'prediction_window': '6months',
            'look_back_durations': ['6months'],
            'train_example_frequency': '1day',
            'test_example_frequency': '3months',
            'test_durations': ['1months'],
            'prediction_frequency': '1d'
        }
        scoring_config = {
            'metric_groups': [
                {'metrics': ['precision@'], 'thresholds': {'top_n': [2]}}
            ],
            'sort_seed': 12345
        }
        grid_config = {
            'sklearn.linear_model.LogisticRegression': {
                'C': [0.00001, 0.0001],
                'penalty': ['l1', 'l2'],
                'random_state': [2193]
            }
        }
        feature_config = [{
            'prefix': 'test_features',
            'from_obj': 'cat_complaints',
            'knowledge_date_column': 'as_of_date',
            'aggregates': [{
                'quantity': 'cat_sightings',
                'metrics': ['count', 'avg'],
            }],
            'intervals': ['1y'],
            'groups': ['entity_id']
        }]
        experiment_config = {
            'events_table': 'events',
            'entity_column_name': 'entity_id',
            'model_comment': 'test2-final-final',
            'feature_aggregations': feature_config,
            'temporal_config': temporal_config,
            'grid_config': grid_config,
            'scoring': scoring_config,
        }

        temp_dir = TemporaryDirectory()
        try:
            pipeline = pipeline_class(
                config=experiment_config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir.name, 'inspections'),
            )

            pipeline.run()

            evaluations = num_linked_evaluations(db_engine)
            assert evaluations > 0

            pipeline = pipeline_class(
                config=experiment_config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir.name, 'inspections'),
                replace=False
            )
            pipeline.make_entity_date_table = Mock()
            pipeline.run()
            assert not pipeline.make_entity_date_table.called
        finally:
            temp_dir.cleanup()


def test_serial_pipeline_reuse():
    reuse_pipeline_test(SerialPipeline)


def test_localparallel_pipeline_reuse():
    reuse_pipeline_test(
        partial(LocalParallelPipeline, n_processes=2, n_db_processes=2)
    )
