import csv
import datetime
import hashlib
import json
import logging
import random
import tempfile

import postgres_copy
import sqlalchemy
from retrying import retry
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import (
    Experiment,
    Matrix,
    Model,
    ExperimentMatrix,
    ExperimentModel
)


def filename_friendly_hash(inputs):
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")

    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True).encode("utf-8")
    ).hexdigest()


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)


DEFAULT_RETRY_KWARGS = {
    "retry_on_exception": retry_if_db_error,
    "wait_exponential_multiplier": 1000,  # wait 2^x*1000ms between each retry
    "stop_max_attempt_number": 14,
    # with this configuration, last wait will be ~2 hours
    # for a total of ~4.5 hours waiting
}


db_retry = retry(**DEFAULT_RETRY_KWARGS)


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


class Batch:
    # modified from
    # http://codereview.stackexchange.com/questions/118883/split-up-an-iterable-into-batches
    def __init__(self, iterable, limit=None):
        self.iterator = iter(iterable)
        self.limit = limit
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def group(self):
        yield self.current
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            self.current = item
            if num == self.limit:
                break
            yield item
        else:
            self.on_going = False

    def __iter__(self):
        while self.on_going:
            yield self.group()


def sort_predictions_and_labels(predictions_proba, labels, sort_seed):
    random.seed(sort_seed)
    predictions_proba_sorted, labels_sorted = zip(
        *sorted(
            zip(predictions_proba, labels),
            key=lambda pair: (pair[0], random.random()),
            reverse=True,
        )
    )
    return predictions_proba_sorted, labels_sorted


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


@db_retry
def save_db_objects(db_engine, db_objects):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (list) SQLAlchemy model objects, corresponding to a valid table
    """
    with tempfile.TemporaryFile(mode="w+") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for db_object in db_objects:
            writer.writerow(
                [getattr(db_object, col.name) for col in db_object.__table__.columns]
            )
        f.seek(0)
        postgres_copy.copy_from(f, type(db_objects[0]), db_engine, format="csv")
