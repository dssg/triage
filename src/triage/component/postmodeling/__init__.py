# coding: utf-8

import os
import yaml
import logging

import sqlalchemy
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import sqlalchemy.dialects.postgresql as postgresql
from sqlalchemy import Column, ForeignKey

from triage.component.catwalk.storage import CSVMatrixStore, HDFMatrixStore, Store, ProjectStorage

logging.basicConfig(level=logging.INFO)


def __db_url():
    environ_url = os.getenv('DATABASE_URL')
    if environ_url:
        logging.info("Getting db connection credentials from DATABASE_URL")
        return environ_url

    if os.path.isfile(os.getenv('DATABASE_FILE')):
        logging.info("Getting db connection credentials from DATABASE_FILE")
        dbfile = open(os.getenv('DATABASE_FILE'))
    else:
        raise EnvironmentError(
            f"could not determine database connection information from "
            f"either process environment (DATABASE_URL) or filesystem "
            f"(DATABASE_FILE)"
        )

    with dbfile:
        dbconfig = yaml.load(dbfile)

    return URL(
        'postgres',
        host=dbconfig['host'],
        username=dbconfig['user'],
        database=dbconfig['db'],
        password=dbconfig['pass'],
        port=dbconfig['port'],
    )


__engine = create_engine(__db_url())
__session = sessionmaker(bind=__engine)
session = __session()

Base = declarative_base()


def get_model(model_id):
    return session.query(Model).get(model_id)


def get_model_group(model_group_id):
    return session.query(ModelGroup).get(model_group_id)


class ModelGroup(Base):
    __tablename__ = 'model_groups'
    __table_args__ = ({"schema": "model_metadata"})

    id = Column('model_group_id', postgresql.INTEGER, primary_key=True)
    type = Column('model_type', postgresql.TEXT)
    features = Column('feature_list', postgresql.ARRAY(postgresql.TEXT))
    config = Column('model_config', postgresql.JSONB)


class Model(Base):
    __tablename__ = 'models'
    __table_args__ = ({"schema": "model_metadata"})

    id = Column('model_id', postgresql.INTEGER, primary_key=True)
    hash = Column('model_hash', postgresql.TEXT)
    run_time = Column(postgresql.TIMESTAMP)
    batch_run_time = Column(postgresql.TIMESTAMP)
    comment = Column('model_comment', postgresql.TEXT)
    experiment = Column('built_by_experiment', postgresql.TEXT)
    train_matrix = Column('train_matrix_uuid', postgresql.TEXT)
    train_end_time = Column(postgresql.TIMESTAMP)
    training_label_timespan = Column(postgresql.INTERVAL)
    size = Column('model_size', postgresql.FLOAT)

    model_group_id = Column('model_group_id', postgresql.INTEGER, ForeignKey('model_metadata.model_groups.model_group_id'))
    model_group = relationship("ModelGroup", backref="models")


class Evaluation(Base):
    __tablename__ = 'evaluations'
    __table_args__ = ({"schema": "test_results"})

    evaluation_start_time = Column(postgresql.TIMESTAMP, primary_key=True)
    evaluation_end_time = Column(postgresql.TIMESTAMP, primary_key=True)
    metric = Column(postgresql.TEXT, primary_key=True)
    parameter = Column(postgresql.TEXT, primary_key=True)

    value = Column(postgresql.FLOAT)
    num_labeled_examples = Column(postgresql.INTEGER)
    num_labeled_above_threshold = Column(postgresql.INTEGER)
    num_positive_labels = Column(postgresql.INTEGER)
    sort_seed =Column(postgresql.INTEGER)
    matrix = Column('matrix_uuid', postgresql.TEXT)

    model_id = Column('model_id', postgresql.INTEGER, ForeignKey('model_metadata.models.model_id'), primary_key=True)
    model = relationship("Model", backref="evaluations")


class Prediction(Base):
    __tablename__ = 'predictions'
    __table_args__ = ({"schema": "test_results"})

    entity_id = Column(postgresql.INTEGER, primary_key=True)
    as_of_date = Column(postgresql.TIMESTAMP, primary_key=True)

    model_id = Column('model_id', postgresql.INTEGER, ForeignKey('model_metadata.models.model_id'), primary_key=True)
    model = relationship("Model", backref="predictions")

    score = Column(postgresql.FLOAT)
    label_value = Column(postgresql.INTEGER)
    test_label_timespan = Column(postgresql.INTERVAL)
    matrix = Column('matrix_uuid', postgresql.TEXT)
