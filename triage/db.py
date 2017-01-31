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
    Table,\
    Text,\
    ForeignKey,\
    MetaData,\
    DDL,\
    event
from sqlalchemy.types import ARRAY
from sqlalchemy.orm import mapper, relationship

metadata = MetaData(schema='results')
event.listen(
    metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS results")
)

class TriageModelBase(object):
    def __init__(self, *args, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class ModelGroup(TriageModelBase):
    pass


class Model(TriageModelBase):
    pass


class FeatureImportance(TriageModelBase):
    pass


class Prediction(TriageModelBase):
    pass


def define_models():
    cols = [
        Column('model_id', Integer, primary_key=True),
        Column('model_group_id', Integer, ForeignKey('model_groups.model_group_id')),
        Column('model_hash', String, unique=True, index=True),
        Column('run_time', DateTime),
        Column('batch_run_time', DateTime),
        Column('model_type', String),
        Column('model_parameters', JSON),
        Column('model_comment', Text),
        Column('batch_comment', Text),
        Column('config', JSON),
        Column('test', Boolean),
    ]
    mapper(
        Model,
        Table('models', metadata, *cols)
    )

def define_feature_importances():
    cols = [
        Column('model_id', Integer, ForeignKey('models.model_id'), primary_key=True),
        Column('feature', String, primary_key=True),
        Column('feature_importance', Numeric),
    ]
    mapper(
        FeatureImportance,
        Table('feature_importances', metadata, *cols),
        properties={'model': relationship(Model)}
    )


def define_model_groups():
    cols = [
        Column('model_group_id', Integer, primary_key=True),
        Column('model_type', String),
        Column('model_parameters', JSON),
        Column('prediction_window', String),
        Column('feature_list', ARRAY(String)),
    ]
    mapper(
        ModelGroup,
        Table('model_groups', metadata, *cols)
    )


def define_predictions(entity_column_name):
    prediction_columns = [
        Column('model_id', Integer, ForeignKey('models.model_id'), primary_key=True),
        Column(entity_column_name, BigInteger, primary_key=True),
        Column('as_of_date', Date, primary_key=True),
        Column('entity_score', Numeric),
        Column('label_value', Integer),
        Column('rank_abs', Integer),
        Column('rank_pct', Float)
    ]
    t = Table('predictions', metadata, *prediction_columns)
    mapper(Prediction, t)


def ensure_db(engine, entity_column_name=None):
    define_predictions(entity_column_name or 'entity_id')
    define_models()
    define_model_groups()
    define_feature_importances()
    metadata.create_all(engine)
