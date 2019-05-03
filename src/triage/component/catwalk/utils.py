import csv
import datetime
import hashlib
import numpy
import json
import logging
import random
from itertools import chain
from functools import partial

import postgres_copy
import sqlalchemy
from retrying import retry
from sqlalchemy.orm import sessionmaker
from ohio import PipeTextIO

from triage.component.results_schema import (
    Experiment,
    Matrix,
    Model,
    ExperimentMatrix,
    ExperimentModel,
)


def filename_friendly_hash(inputs):
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")

    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_subset_table_name(subset_config):
    return "subset_{}_{}".format(
        subset_config.get("name", "default"),
        filename_friendly_hash(subset_config),
    )


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


AVAILABLE_TIEBREAKERS = {'random', 'best', 'worst'}

def sort_predictions_and_labels(predictions_proba, labels, tiebreaker='random', sort_seed=None, parallel_arrays=()):
    """Sort predictions and labels with a configured tiebreaking rule

    Args:
        predictions_proba (numpy.array) The predicted scores
        labels (numpy.array) The numeric labels (1/0, not True/False)
        tiebreaker (string) The tiebreaking method ('best', 'worst', 'random')
        sort_seed (signed int) The sort seed. Needed if 'random' tiebreaking is picked.
        parallel_arrays (tuple of numpy.array) Any other arrays, understood to be the same size
            as the predictions and labels, that should be sorted alongside them.

    Returns:
        (tuple) (predictions_proba, labels), sorted
    """
    if len(labels) == 0:
        logging.debug("No labels present, skipping sorting.")
        if parallel_arrays:
            return (predictions_proba, labels, parallel_arrays)
        else:
            return (predictions_proba, labels)
    mask = None
    if tiebreaker == 'random':
        if not sort_seed:
            raise ValueError("If random tiebreaker is used, a sort seed must be given")
        random.seed(sort_seed)
        numpy.random.seed(sort_seed)
        random_arr = numpy.random.rand(*predictions_proba.shape)
        mask = numpy.lexsort((random_arr, predictions_proba))
    elif tiebreaker == 'worst':
        mask = numpy.lexsort((-labels, predictions_proba))
    elif tiebreaker == 'best':
        mask = numpy.lexsort((labels, predictions_proba))
    else:
        raise ValueError("Unknown tiebreaker")

    return_value = [
        numpy.flip(predictions_proba[mask]),
        numpy.flip(labels[mask]),
    ]
    if parallel_arrays:
        return_value.append(tuple(numpy.flip(arr[mask]) for arr in parallel_arrays))
    return return_value

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


def _write_csv(file_like, db_objects, type_of_object):
    writer = csv.writer(file_like, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for db_object in db_objects:
        if type(db_object) != type_of_object:
            raise TypeError("Cannot copy collection of objects to db as they are not all "
                            f"of the same type. First object was {type_of_object} "
                            f"and later encountered a {type(db_object)}")
        writer.writerow(
            [getattr(db_object, col.name) for col in db_object.__table__.columns]
        )


@db_retry
def save_db_objects(db_engine, db_objects):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (iterable) SQLAlchemy model objects, corresponding to a valid table
    """
    db_objects = iter(db_objects)
    first_object = next(db_objects)
    type_of_object = type(first_object)

    with PipeTextIO(partial(
            _write_csv,
            db_objects=chain((first_object,), db_objects),
            type_of_object=type_of_object
    )) as pipe:
        postgres_copy.copy_from(pipe, type_of_object, db_engine, format="csv")
