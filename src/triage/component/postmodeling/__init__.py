# coding: utf-8

import os
import yaml
import logging

import sqlalchemy
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
import sqlalchemy.dialects.postgresql as postgresql
from sqlalchemy import Column, ForeignKey

import pandas as pd

from triage.component.catwalk.storage import CSVMatrixStore, Store, ProjectStorage
from triage.component.results_schema.schema import TestEvaluation, TestPrediction, Model, ModelGroup

logging.basicConfig(level=logging.INFO)


def _db_url_from_environment():
    environ_url = os.getenv('DATABASE_URL')
    if environ_url:
        logging.info("Getting db connection credentials from DATABASE_URL")
        return environ_url

    dbfile_path = os.getenv('DATABASE_FILE')
    if dbfile_path:
        if not os.path.isfile(dbfile_path):
            logging.error('No such database file path: %s', dbfile_path)
            return None
    else:
        logging.warn("Neither environment variable DATABASE_URL or DATABASE_FILE set")
        return None

    with open(dbfile_path) as dbfile:
        dbconfig = yaml.load(dbfile)

    return URL(
        'postgres',
        host=dbconfig['host'],
        username=dbconfig['user'],
        database=dbconfig['db'],
        password=dbconfig['pass'],
        port=dbconfig['port'],
    )


def create_session(engine=None):
    if engine is None:
        url = _db_url_from_environment()
        if not url:
            return None

        engine = create_engine(url)

    return Session(bind=engine)


Base = declarative_base()

def get_model(model_id: int, session=None) -> Model:
    if session is None:
        session = create_session()
    return session.query(Model).get(model_id)


def get_model_group(model_group_id: int, session=None) -> ModelGroup:
    if session is None:
        session = create_session()
    return session.query(ModelGroup).get(model_group_id)


def get_predictions(model: Model) -> TestPrediction:
    predictions = pd.DataFrame([prediction.__dict__ for prediction in model.predictions])
    return predictions.drop('_sa_instance_state', axis=1).set_index(['as_of_date', 'entity_id'])


def get_evaluations(model: Model) -> TestEvaluation:
    evaluations =  pd.DataFrame([evaluation.__dict__ for evaluation in model.evaluations])
    evaluations['model_group_id'] = model.model_group_id
    return evaluations.drop('_sa_instance_state', axis=1).set_index(['model_group_id', 'model_id', 'metric', 'parameter'])
