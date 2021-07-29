from triage.component.results_schema import RetrainModel, Retrain
from triage.component.catwalk.utils import db_retry, filename_friendly_hash

import re
from sqlalchemy.orm import sessionmaker
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
    join triage_metadata.models on (experiments.experiment_hash = models.built_by_experiment)
    where model_id = %s
    '''
    (config,) = db_engine.execute(get_experiment_query, model_id).first()
    return config


def experiment_config_from_model_group_id(db_engine, model_group_id):
    """Get original experiment config from model_id 
    Args:
            db_engine (sqlalchemy.db.engine)
            model_id (int) The id of a given model in the database

    Returns: (dict) experiment config
    """
    get_experiment_query = '''
    select experiment_runs.id as run_id, experiments.config
    from triage_metadata.experiments
    join triage_metadata.models 
    on (experiments.experiment_hash = models.built_by_experiment)
    join triage_metadata.experiment_runs 
    on (experiment_runs.id = models.built_in_triage_run)
    where model_group_id = %s
    order by experiment_runs.start_time desc
    '''
    (run_id, config) = db_engine.execute(get_experiment_query, model_group_id).first()
    return run_id, config


def get_model_group_info(db_engine, model_group_id):
    query = """
    SELECT model_group_id, model_type, hyperparameters, model_id as model_id_last_split
    FROM triage_metadata.models
    WHERE model_group_id = %s
    ORDER BY train_end_time DESC
    """
    model_group_info = db_engine.execute(query, model_group_id).fetchone()
    return dict(model_group_info)


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
    feature_names_in_group = [f for f in matrix_metadata['feature_names'] if re.match(f'\\A{feature_prefix}_', f)]
    logger.spam("Feature names in group = %s", feature_names_in_group)
    
    return feature_group, feature_names_in_group


def get_feature_needs_imputation_in_train(aggregation, feature_names):
    """Returns features that needs imputation from training data
    Args:
        aggregation (SpacetimeAggregation)
        feature_names (list) A list of feature names
    """
    features_imputed_in_train = [
        f for f in set(feature_names)
        if not f.endswith('_imp') 
        and aggregation.imputation_flag_base(f) + '_imp' in feature_names
    ]
    logger.spam("Features imputed in train = %s", features_imputed_in_train)
    return features_imputed_in_train


def get_feature_needs_imputation_in_production(aggregation, db_engine):
    """Returns features that needs imputation from triage_production
    Args:
        aggregation (SpacetimeAggregation)
        db_engine (sqlalchemy.db.engine)
    """
    with db_engine.begin() as conn:
        nulls_results = conn.execute(aggregation.find_nulls())
    
    null_counts = nulls_results.first().items()
    features_imputed_in_production = [col for (col, val) in null_counts if val is not None and val > 0]
    
    return features_imputed_in_production


@db_retry
def associate_models_with_retrain(retrain_hash, model_hashes, db_engine):
    session = sessionmaker(bind=db_engine)()
    for model_hash in model_hashes:
        session.merge(RetrainModel(retrain_hash=retrain_hash, model_hash=model_hash))
    session.commit()
    session.close()
    logger.spam("Associated models with retrain in database")

@db_retry
def save_retrain_and_get_hash(config, db_engine):
    retrain_hash = filename_friendly_hash(config)
    session = sessionmaker(bind=db_engine)()
    session.merge(Retrain(retrain_hash=retrain_hash, config=config))
    session.commit()
    session.close()
    return retrain_hash


