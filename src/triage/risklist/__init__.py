from triage.component.results_schema import upgrade_db, Experiment, ExperimentModel 
from triage.component.architect.entity_date_table_generators import EntityDateTableGenerator, DEFAULT_ACTIVE_STATE
from triage.component.architect.features import FeatureGenerator
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.utils import filename_friendly_hash
from triage.util.conf import dt_from_str
from triage.util.db import scoped_session
from sqlalchemy import select

from collections import OrderedDict
import json
import re

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)


def experiment_config_from_model_id(db_engine, model_id):
    """Get original experiment config from model_id 
    Args:
            db_engine (sqlalchemy.db.engine)
            model_id (int) The id of a given model in the database

    Returns: (dict) experiment config
    """
    get_experiment_query = '''select experiments.config
    from triage_metadata.experiments
    join triage_metadata.experiment_models using (experiment_hash)
    join triage_metadata.models using (model_hash)
    where model_id = %s
    '''
    (config,) = db_engine.execute(get_experiment_query, model_id).first()
    return config
    

def train_matrix_info_from_model_id(db_engine, model_id):
    """Get original train matrix information from model_id 
    Args:
            db_engine (sqlalchemy.db.engine)
            model_id (int) The id of a given model in the database

    Returns: (str, dict) matrix uuid and matrix metadata
    """
    get_train_matrix_query = """
        select matrix_uuid, matrices.matrix_metadata
        from triage_metadata.matrices
        join triage_metadata.models on (models.train_matrix_uuid = matrices.matrix_uuid)
        where model_id = %s
    """
    return db_engine.execute(get_train_matrix_query, model_id).first()


def get_feature_names(aggregation, matrix_metadata):
    """Returns a feature group name and a list of feature names from a SpacetimeAggregation object"""
    feature_prefix = aggregation.prefix
    logger.spam("Feature prefix = %s", feature_prefix)
    feature_group = aggregation.get_table_name(imputed=True).split('.')[1].replace('"', '')
    logger.spam("Feature group = %s", feature_group)
    feature_names_in_group = [f for f in matrix_metadata['feature_names'] if re.match(f'\\A{feature_prefix}', f)]
    logger.spam("Feature names in group = %s", feature_names_in_group)
    
    return feature_group, feature_names_in_group

def get_feature_needs_imputation_in_train(feature_names):
    features_imputed_in_train = [
                f for f in set(feature_names)
                if not f.endswith('_imp') 
                and '_'.join(f.split('_')[0:-1]) + '_imp' in feature_names
            ]
    logger.spam("Features imputed in train = %s", features_imputed_in_train)
    return features_imputed_in_train


def get_feature_needs_imputation_in_production(aggregation, conn):
    nulls_results = conn.execute(aggregation.find_nulls())
    null_counts = nulls_results.first().items()
    features_imputed_in_production = [col for (col, val) in null_counts if val > 0]
    
    return features_imputed_in_production


def generate_risk_list(db_engine, project_storage, model_id, as_of_date):
    """Generate the risk list based model_id and as_of_date

    Args:
            db_engine (sqlalchemy.db.engine)
            project_storage (catwalk.storage.ProjectStorage)
            model_id (int) The id of a given model in the database
            as_of_date (string) a date string like "YYYY-MM-DD"
    """
    logger.spam("In RISK LIST................")
    upgrade_db(db_engine=db_engine)
    matrix_storage_engine = project_storage.matrix_storage_engine()
    # 1. Get feature and cohort config from database
    (train_matrix_uuid, matrix_metadata) = train_matrix_info_from_model_id(db_engine, model_id)
    experiment_config = experiment_config_from_model_id(db_engine, model_id)

    # 2. Generate cohort
    cohort_table_name = f"production.cohort_{experiment_config['cohort_config']['name']}"
    cohort_table_generator = EntityDateTableGenerator(
        db_engine=db_engine,
        query=experiment_config['cohort_config']['query'],
        entity_date_table_name=cohort_table_name
    )
    cohort_table_generator.generate_entity_date_table(as_of_dates=[dt_from_str(as_of_date)])

    # 3. Generate feature aggregations
    feature_generator = FeatureGenerator(
        db_engine=db_engine,
        features_schema_name="production",
        feature_start_time=experiment_config['temporal_config']['feature_start_time'],
    )
    collate_aggregations = feature_generator.aggregations(
        feature_aggregation_config=experiment_config['feature_aggregations'],
        feature_dates=[as_of_date],
        state_table=cohort_table_name
    )
    feature_generator.process_table_tasks(
        feature_generator.generate_all_table_tasks(
            collate_aggregations,
            task_type='aggregation'
        )
    )

    # 4. Reconstruct feature disctionary from feature_names and generate imputation
    reconstructed_feature_dictionary = {}
    imputation_table_tasks = OrderedDict()
    with db_engine.begin() as conn:
        for aggregation in collate_aggregations:
            feature_group, feature_names = get_feature_names(aggregation, matrix_metadata)
            reconstructed_feature_dictionary[feature_group] = feature_names

            # Make sure that the features imputed in training should also be imputed in production
            
            features_imputed_in_train = get_feature_needs_imputation_in_train(feature_names)
            features_imputed_in_production = get_feature_needs_imputation_in_production(aggregation, conn)

            total_impute_cols = set(features_imputed_in_production) | set(features_imputed_in_train)
            total_nonimpute_cols = set(f for f in set(feature_names) if '_imp' not in f) - total_impute_cols
            
            task_generator = feature_generator._generate_imp_table_tasks_for
            
            imputation_table_tasks.update(task_generator(
                aggregation,
                impute_cols=list(total_impute_cols),
                nonimpute_cols=list(total_nonimpute_cols)
                )
            )
    feature_generator.process_table_tasks(imputation_table_tasks)

    # 5. Build matrix
    db_config = {
        "features_schema_name": "production",
        "labels_schema_name": "public",
        "cohort_table_name": cohort_table_name,
    }

    matrix_builder = MatrixBuilder(
        db_config=db_config,
        matrix_storage_engine=matrix_storage_engine,
                engine=db_engine,
        experiment_hash=None,
        replace=True,
    )

    matrix_metadata = {
        'as_of_times': [as_of_date],
        'matrix_id': str(as_of_date) + '_prediction',
        'state': DEFAULT_ACTIVE_STATE,
        'test_duration': '1y',
        'matrix_type': 'production',
        'label_timespan': None,
        'label_name': experiment_config['label_config']['name'],
        'indices': ["entity_id", "as_of_date"],
        'feature_start_time': experiment_config['temporal_config']['feature_start_time'],
    }

    matrix_uuid = filename_friendly_hash(matrix_metadata)

    matrix_builder.build_matrix(
        as_of_times=[as_of_date],
        label_name=experiment_config['label_config']['name'],
        label_type=None,
        feature_dictionary=reconstructed_feature_dictionary,
        matrix_metadata=matrix_metadata,
        matrix_uuid=matrix_uuid,
        matrix_type="production",
    )

    # 6. Predict the risk score for production
    predictor = Predictor(
        model_storage_engine=project_storage.model_storage_engine(),
        db_engine=db_engine,
        rank_order='best'
    )

    predictor.predict(
        model_id=model_id,
        matrix_store=matrix_storage_engine.get_store(matrix_uuid),
        misc_db_parameters={},
        train_matrix_columns=matrix_storage_engine.get_store(train_matrix_uuid).columns()
    )
