from sqlalchemy import \
    Column,\
    BigInteger,\
    Boolean,\
    Integer,\
    Interval,\
    String,\
    Numeric,\
    DateTime,\
    JSON,\
    Float,\
    Text,\
    ForeignKey,\
    MetaData,\
    DDL,\
    create_engine,\
    event
from sqlalchemy.types import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.url import URL

import os
import yaml


Base = declarative_base(metadata=MetaData(schema='results'))
event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS results")
)

group_proc_filename = os.path.join(
    os.path.dirname(__file__),
    'model_group_stored_procedure.sql'
)
with open(group_proc_filename) as f:
    stmt = f.read()

event.listen(
    Base.metadata,
    'before_create',
    DDL(stmt)
)


class ModelGroup(Base):
    __tablename__ = 'model_groups'
    model_group_id = Column(Integer, primary_key=True)
    model_type = Column(Text)
    model_parameters = Column(JSONB)
    prediction_window = Column(Text)
    feature_list = Column(ARRAY(Text))


class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    model_group_id = Column(Integer, ForeignKey('model_groups.model_group_id'))
    model_hash = Column(String, unique=True, index=True)
    run_time = Column(DateTime)
    batch_run_time = Column(DateTime)
    model_type = Column(String)
    model_parameters = Column(JSONB)
    model_comment = Column(Text)
    batch_comment = Column(Text)
    config = Column(JSON)
    test = Column(Boolean)
    train_matrix_uuid = Column(Text)

    def delete(self, session):
        # basically implement a cascade, in case cascade is not implemented
        session.query(FeatureImportance)\
            .filter_by(model_id=self.model_id)\
            .delete()
        session.query(Evaluation)\
            .filter_by(model_id=self.model_id)\
            .delete()
        session.query(Prediction)\
            .filter_by(model_id=self.model_id)\
            .delete()
        session.delete(self)


class FeatureImportance(Base):
    __tablename__ = 'feature_importances'
    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    model = relationship(Model)
    feature = Column(String, primary_key=True)
    feature_importance = Column(Numeric)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)


class Prediction(Base):
    __tablename__ = 'predictions'
    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric)
    label_value = Column(Integer)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)
    matrix_uuid = Column(Text)


class Evaluation(Base):
    __tablename__ = 'evaluations'
    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    evaluation_start_time = Column(DateTime, primary_key=True)
    evaluation_end_time = Column(DateTime, primary_key=True)
    prediction_frequency = Column(Interval, primary_key=True)
    metric = Column(String, primary_key=True)
    parameter = Column(String, primary_key=True)
    value = Column(Numeric)


def ensure_db(engine):
    Base.metadata.create_all(engine)


def connect(poolclass=NullPool):
    with open('database.yaml') as f:
        profile = yaml.load(f)
        dbconfig = {
            'host': profile['host'],
            'username': profile['user'],
            'database': profile['db'],
            'password': profile['pass'],
            'port': profile['port'],
        }
        dburl = URL('postgres', **dbconfig)
        return create_engine(dburl, poolclass=poolclass)
