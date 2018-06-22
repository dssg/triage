import os

from sqlalchemy import (
    Column,
    Boolean,
    Integer,
    Text,
    DateTime,
    Interval,
    Float,
    DDL,
    event
)

from sqlalchemy.ext.declarative import declarative_base

DEFAULT_QUERY_LOC = "../../import_tool/triage_default_queries/"
DEFAULT_SQL_FUNCTIONS = ['get_prediction_lags_daily.sql',
                         'get_prediction_lags_hist.sql']

Base = declarative_base()

event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS model_monitor;")
)


class ModelGroupParameters(Base):
    __tablename__ = 'model_group_parameters'
    __table_args__ = {'schema': 'model_monitor'}

    model_group_id = Column(Integer, primary_key=True)
    model_parameter_type = Column(Text, primary_key=True)
    model_parameter = Column(Text, primary_key=True)
    model_parameter_value = Column(Text)


class ModelParameters(Base):
    __tablename__ = 'model_parameters'
    __table_args__ = {'schema': 'model_monitor'}

    model_id = Column(Integer, primary_key=True)
    model_parameter_type = Column(Text, primary_key=True)
    model_parameter = Column(Text, primary_key=True)
    model_parameter_value = Column(Text)


class PredictionMetricDefs(Base):
    __tablename__ = 'prediction_metric_defs'
    __table_args = {'schema': 'model_monitor'}

    prediction_metric_id = Column(Integer, primary_key=True)
    metric_type = Column(Text, nullable=False)
    threshold = Column(Float)
    use_top_entities = Column(Boolean, nullable=False, default=True)
    use_lag_as_reference = Column(Boolean, nullable=False, default=False)
    compare_interval = Column(Interval, nullable=False)


class PredictionMetrics(Base):
    __tablename__ = 'prediction_metrics'
    __table_args__ = {'schema': 'model_monitor'}

    prediction_metric_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    metric_value = Column(Float)


class FeatureImportanceMetricDefs(Base):
    __tablename__ = 'feature_metric_defs'
    __table_args = {'schema': 'model_monitor'}

    feature_metric_id = Column(Integer, primary_key=True)
    metric_type = Column(Text, nullable=False)
    threshold = Column(Float)
    use_top_entities = Column(Boolean, nullable=False, default=True)
    use_lag_as_reference = Column(Boolean, nullable=False, default=False)
    compare_interval = Column(Interval, nullable=False)


class FeatureImportanceMetrics(Base):
    __tablename__ = 'feature_metrics'
    __table_args__ = {'schema': 'model_monitor'}

    feature_metric_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    feature = Column(Text, primary_key=True)
    metric_value = Column(Float)
