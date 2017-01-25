from sqlalchemy import \
    Column,\
    Integer,\
    String,\
    Numeric,\
    DateTime,\
    JSON,\
    ForeignKey,\
    MetaData,\
    DDL,\
    event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base(metadata=MetaData(schema='results'))
event.listen(Base.metadata, 'before_create', DDL("CREATE SCHEMA IF NOT EXISTS results"))


class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    unique_identifier = Column(String, unique=True, index=True)
    run_time = Column(DateTime)
    model_type = Column(String)
    model_parameters = Column(JSON)


class FeatureImportance(Base):
    __tablename__ = 'feature_importances'
    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    model = relationship(Model)
    feature = Column(String, primary_key=True)
    feature_importance = Column(Numeric)


def ensure_db(engine):
    Base.metadata.create_all(engine)
