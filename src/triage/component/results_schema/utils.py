import logging
from sqlalchemy.orm import sessionmaker
from triage.util.db import db_retry
from triage.util.has import filename_friendly_hash

from .schema import (
    Experiment,
    Matrix,
    Model,
    ExperimentMatrix,
    ExperimentModel,
)


@db_retry
def save_experiment_and_get_hash(config, db_engine):
    experiment_hash = filename_friendly_hash(config)
    session = sessionmaker(bind=db_engine)()
    session.merge(Experiment(experiment_hash=experiment_hash, config=config))
    session.commit()
    session.close()
    return experiment_hash


@db_retry
def associate_matrices_with_experiment(experiment_hash, matrix_uuids, db_engine):
    session = sessionmaker(bind=db_engine)()
    for matrix_uuid in matrix_uuids:
        session.merge(ExperimentMatrix(experiment_hash=experiment_hash, matrix_uuid=matrix_uuid))
    session.commit()
    session.close()
    logging.info("Associated matrices with experiment in database")


@db_retry
def associate_models_with_experiment(experiment_hash, model_hashes, db_engine):
    session = sessionmaker(bind=db_engine)()
    for model_hash in model_hashes:
        session.merge(ExperimentModel(experiment_hash=experiment_hash, model_hash=model_hash))
    session.commit()
    session.close()
    logging.info("Associated models with experiment in database")


@db_retry
def missing_matrix_uuids(experiment_hash, db_engine):
    """Compare the contents of the experiment_matrices table with that of the
    matrices table to produce a list of matrix_uuids that the experiment wants
    but are not available.
    """
    query = f"""
        select experiment_matrices.matrix_uuid
        from {ExperimentMatrix.__table__.fullname} experiment_matrices
        left join {Matrix.__table__.fullname} matrices
        on (experiment_matrices.matrix_uuid = matrices.matrix_uuid)
        where experiment_hash = %s
        and matrices.matrix_uuid is null
    """
    return [row[0] for row in db_engine.execute(query, experiment_hash)]


@db_retry
def missing_model_hashes(experiment_hash, db_engine):
    """Compare the contents of the experiment_models table with that of the
    models table to produce a list of model hashes the experiment wants
    but are not available.
    """
    query = f"""
        select experiment_models.model_hash
        from {ExperimentModel.__table__.fullname} experiment_models
        left join {Model.__table__.fullname} models
        on (experiment_models.model_hash = models.model_hash)
        where experiment_hash = %s
        and models.model_hash is null
    """
    return [row[0] for row in db_engine.execute(query, experiment_hash)]


@db_retry
def retrieve_model_id_from_hash(db_engine, model_hash):
    """Retrieves a model id from the database that matches the given hash

    Args:
        db_engine (sqlalchemy.engine) A database engine
        model_hash (str) The model hash to lookup

    Returns: (int) The model id (if found in DB), None (if not)
    """
    session = sessionmaker(bind=db_engine)()
    try:
        saved = session.query(Model).filter_by(model_hash=model_hash).one_or_none()
        return saved.model_id if saved else None
    finally:
        session.close()


@db_retry
def retrieve_model_hash_from_id(db_engine, model_id):
    """Retrieves the model hash associated with a given model id

    Args:
        model_id (int) The id of a given model in the database

    Returns: (str) the stored hash of the model
    """
    session = sessionmaker(bind=db_engine)()
    try:
        return session.query(Model).get(model_id).model_hash
    finally:
        session.close()
