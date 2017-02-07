from sqlalchemy import \
    Column,\
    BigInteger,\
    Boolean,\
    Integer,\
    String,\
    Numeric,\
    Date,\
    DateTime,\
    JSON,\
    Float,\
    Text,\
    ForeignKey,\
    MetaData,\
    DDL,\
    event
from sqlalchemy.types import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base(metadata=MetaData(schema='results'))
event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS results")
)


class ModelGroup(Base):
    __tablename__ = 'model_groups'
    model_group_id = Column(Integer, primary_key=True)
    model_type = Column(String)
    model_parameters = Column(JSON)
    prediction_window = Column(String)
    feature_list = Column(ARRAY(String))


class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    model_group_id = Column(Integer, ForeignKey('model_groups.model_group_id'))
    model_hash = Column(String, unique=True, index=True)
    run_time = Column(DateTime)
    batch_run_time = Column(DateTime)
    model_type = Column(String)
    model_parameters = Column(JSON)
    model_comment = Column(Text)
    batch_comment = Column(Text)
    config = Column(JSON)
    test = Column(Boolean)
    train_matrix_uuid = Column(Text)


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
    unit_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    unit_score = Column(Numeric)
    label_value = Column(Integer)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)
    matrix_uuid = Column(Text)


def ensure_db(engine):
    Base.metadata.create_all(engine)
