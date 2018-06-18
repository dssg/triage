import os
from os.path import isfile, join
from tempfile import TemporaryDirectory

import testing.postgresql
from triage import create_engine

from triage.component.catwalk.db import ensure_db
from triage.component.catwalk.storage import FSModelStorageEngine

from tests.utils import sample_config, populate_source_data

from triage.experiments import SingleThreadedExperiment
from triage.database_reflection import schema_tables
from triage.validation_primitives import table_should_have_data


import logging
logging.basicConfig(level=logging.INFO)

def test_get_splits():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            assert experiment.split_definitions


def test_cohort():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'cohort_config': sample_config()['cohort_config'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            table_should_have_data(experiment.sparse_states_table_name, db_engine)


def test_labels():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'label_config': sample_config()['label_config'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            table_should_have_data(experiment.labels_table_name, db_engine)


def test_preimputation_features():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'feature_aggregations': sample_config()['feature_aggregations'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            generated_tables = [
                table
                for table in schema_tables(experiment.features_schema_name, db_engine).keys()
                if '_aggregation' in table
            ]

            assert len(generated_tables) == len(sample_config()['feature_aggregations'])
            for table in generated_tables:
                table_should_have_data(table, db_engine)


def test_postimputation_features():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'feature_aggregations': sample_config()['feature_aggregations'],
            'cohort_config': sample_config()['cohort_config'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            generated_tables = [
                table
                for table in schema_tables(experiment.features_schema_name, db_engine).keys()
                if '_aggregation_imputed' in table
            ]

            assert len(generated_tables) == len(sample_config()['feature_aggregations'])
            for table in generated_tables:
                table_should_have_data(table, db_engine)


def test_generate_matrices():
    with testing.postgresql.Postgresql() as postgresql:
        config = {
            'temporal_config': sample_config()['temporal_config'],
            'feature_aggregations': sample_config()['feature_aggregations'],
            'cohort_config': sample_config()['cohort_config'],
            'label_config': sample_config()['label_config'],
            'config_version': sample_config()['config_version']
        }
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        with TemporaryDirectory() as temp_dir:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=db_engine,
                model_storage_class=FSModelStorageEngine,
                project_path=os.path.join(temp_dir, 'inspections'),
                cleanup=False
            )
            experiment.run()
            matrices_path = os.path.join(temp_dir, 'inspections', 'matrices')
            matrices_and_metadata = [f for f in os.listdir(matrices_path) if isfile(join(matrices_path, f))]
            matrices = experiment.matrix_build_tasks
            assert len(matrices) > 0
            for matrix in matrices:
                assert '{}.csv'.format(matrix) in matrices_and_metadata
                assert '{}.yaml'.format(matrix) in matrices_and_metadata
