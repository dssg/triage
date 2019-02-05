import os.path

from sqlalchemy import (
    Column,
    BigInteger,
    Boolean,
    Integer,
    Interval,
    String,
    Numeric,
    DateTime,
    JSON,
    Float,
    Text,
    ForeignKey,
    DDL,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import ARRAY
from sqlalchemy.sql import func

# One declarative_base object for each schema created
Base = declarative_base()

schemas = (
    "CREATE SCHEMA IF NOT EXISTS model_metadata;"
    " CREATE SCHEMA IF NOT EXISTS test_results;"
    " CREATE SCHEMA IF NOT EXISTS train_results;"
)

event.listen(Base.metadata, "before_create", DDL(schemas))

group_proc_filename = os.path.join(
    os.path.dirname(__file__), "model_group_stored_procedure.sql"
)
with open(group_proc_filename) as fd:
    stmt = fd.read()

event.listen(Base.metadata, "before_create", DDL(stmt))


class Experiment(Base):

    __tablename__ = "experiments"
    __table_args__ = {"schema": "model_metadata"}

    experiment_hash = Column(String, primary_key=True)
    config = Column(JSONB)


class ModelGroup(Base):

    __tablename__ = "model_groups"
    __table_args__ = {"schema": "model_metadata"}

    model_group_id = Column(Integer, primary_key=True)
    model_type = Column(Text)
    hyperparameters = Column(JSONB)
    feature_list = Column(ARRAY(Text))
    model_config = Column(JSONB)


class ListPrediction(Base):

    __tablename__ = "list_predictions"
    __table_args__ = {"schema": "model_metadata"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)
    matrix_uuid = Column(Text)
    test_label_timespan = Column(Interval)

    model_rel = relationship("Model")


class ExperimentMatrix(Base):
    __tablename__ = "experiment_matrices"
    __table_args__ = {"schema": "model_metadata"}

    experiment_hash = Column(
        String,
        ForeignKey("model_metadata.experiments.experiment_hash"),
        primary_key=True
    )

    matrix_uuid = Column(String, primary_key=True)


class Matrix(Base):

    __tablename__ = "matrices"
    __table_args__ = {"schema": "model_metadata"}

    matrix_id = Column(String)
    matrix_uuid = Column(String, unique=True, index=True, primary_key=True)
    matrix_type = Column(String)  # 'train' or 'test'
    labeling_window = Column(Interval)
    num_observations = Column(Integer)
    creation_time = Column(DateTime(timezone=True), server_default=func.now())
    lookback_duration = Column(Interval)
    feature_start_time = Column(DateTime)
    matrix_metadata = Column(JSONB)
    built_by_experiment = Column(
        String, ForeignKey("model_metadata.experiments.experiment_hash")
    )


class Model(Base):

    __tablename__ = "models"
    __table_args__ = {"schema": "model_metadata"}

    model_id = Column(Integer, primary_key=True)
    model_group_id = Column(
        Integer, ForeignKey("model_metadata.model_groups.model_group_id")
    )
    model_hash = Column(String, unique=True, index=True)
    run_time = Column(DateTime)
    batch_run_time = Column(DateTime)
    model_type = Column(String)
    hyperparameters = Column(JSONB)
    model_comment = Column(Text)
    batch_comment = Column(Text)
    config = Column(JSON)
    built_by_experiment = Column(
        String, ForeignKey("model_metadata.experiments.experiment_hash")
    )
    train_end_time = Column(DateTime)
    test = Column(Boolean)
    train_matrix_uuid = Column(Text, ForeignKey("model_metadata.matrices.matrix_uuid"))
    training_label_timespan = Column(Interval)
    model_size = Column(Float)

    model_group_rel = relationship("ModelGroup")
    matrix_rel = relationship("Matrix")

    def delete(self, session):
        # basically implement a cascade, in case cascade is not implemented
        (session.query(FeatureImportance).filter_by(model_id=self.model_id).delete())
        (session.query(TestEvaluation).filter_by(model_id=self.model_id).delete())
        (session.query(TestPrediction).filter_by(model_id=self.model_id).delete())
        session.delete(self)


class ExperimentModel(Base):
    __tablename__ = "experiment_models"
    __table_args__ = {"schema": "model_metadata"}

    experiment_hash = Column(
        String,
        ForeignKey("model_metadata.experiments.experiment_hash"),
        primary_key=True
    )
    model_hash = Column(String, primary_key=True)

    model_rel = relationship("Model", primaryjoin=(Model.model_hash == model_hash), foreign_keys=model_hash)
    experiment_rel = relationship("Experiment")


class FeatureImportance(Base):

    __tablename__ = "feature_importances"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    model = relationship(Model)
    feature = Column(String, primary_key=True)
    feature_importance = Column(Numeric)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)

    model_rel = relationship("Model")


class TestPrediction(Base):

    __tablename__ = "predictions"
    __table_args__ = {"schema": "test_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric)
    label_value = Column(Integer)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)
    matrix_uuid = Column(Text, ForeignKey("model_metadata.matrices.matrix_uuid"))
    test_label_timespan = Column(Interval)

    model_rel = relationship("Model")


class TrainPrediction(Base):

    __tablename__ = "predictions"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric)
    label_value = Column(Integer)
    rank_abs = Column(Integer)
    rank_pct = Column(Float)
    matrix_uuid = Column(Text, ForeignKey("model_metadata.matrices.matrix_uuid"))
    test_label_timespan = Column(Interval)

    model_rel = relationship("Model")


class IndividualImportance(Base):

    __tablename__ = "individual_importances"
    __table_args__ = {"schema": "test_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    feature = Column(String, primary_key=True)
    method = Column(String, primary_key=True)
    feature_value = Column(Float)
    importance_score = Column(Float)

    model_rel = relationship("Model")


class TestEvaluation(Base):

    __tablename__ = "evaluations"
    __table_args__ = {"schema": "test_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    evaluation_start_time = Column(DateTime, primary_key=True)
    evaluation_end_time = Column(DateTime, primary_key=True)
    as_of_date_frequency = Column(Interval, primary_key=True)
    matrix_uuid = Column(Text, ForeignKey("model_metadata.matrices.matrix_uuid"))
    metric = Column(String, primary_key=True)
    parameter = Column(String, primary_key=True)
    value = Column(Numeric)
    num_labeled_examples = Column(Integer)
    num_labeled_above_threshold = Column(Integer)
    num_positive_labels = Column(Integer)
    sort_seed = Column(Integer)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")


class TrainEvaluation(Base):

    __tablename__ = "evaluations"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("model_metadata.models.model_id"), primary_key=True
    )
    evaluation_start_time = Column(DateTime, primary_key=True)
    evaluation_end_time = Column(DateTime, primary_key=True)
    as_of_date_frequency = Column(Interval, primary_key=True)
    matrix_uuid = Column(Text, ForeignKey("model_metadata.matrices.matrix_uuid"))
    metric = Column(String, primary_key=True)
    parameter = Column(String, primary_key=True)
    value = Column(Numeric)
    num_labeled_examples = Column(Integer)
    num_labeled_above_threshold = Column(Integer)
    num_positive_labels = Column(Integer)
    sort_seed = Column(Integer)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")
