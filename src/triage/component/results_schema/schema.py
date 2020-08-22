import os.path
import enum
import datetime

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
from sqlalchemy.types import ARRAY, Enum
from sqlalchemy.sql import func

# One declarative_base object for each schema created
Base = declarative_base()

schemas = (
    "CREATE SCHEMA IF NOT EXISTS triage_metadata;"
    " CREATE SCHEMA IF NOT EXISTS test_results;"
    " CREATE SCHEMA IF NOT EXISTS train_results;"
)

event.listen(Base.metadata, "before_create", DDL(schemas))

group_proc_filename = os.path.join(
    os.path.dirname(__file__), "sql", "model_group_stored_procedure.sql"
)
with open(group_proc_filename) as fd:
    stmt = fd.read()

event.listen(Base.metadata, "before_create", DDL(stmt))

nuke_triage_filename = os.path.join(
    os.path.dirname(__file__), "sql", "nuke_triage.sql"
)
with open(nuke_triage_filename) as fd:
    stmt = fd.read()


event.listen(Base.metadata, "before_create", DDL(stmt.replace('%', '%%')))


class Experiment(Base):

    __tablename__ = "experiments"
    __table_args__ = {"schema": "triage_metadata"}

    experiment_hash = Column(String, primary_key=True)
    config = Column(JSONB)
    time_splits = Column(Integer)
    as_of_times = Column(Integer)
    feature_blocks = Column(Integer)
    total_features = Column(Integer)
    feature_group_combinations = Column(Integer)
    matrices_needed = Column(Integer)
    grid_size = Column(Integer)
    models_needed = Column(Integer)
    random_seed = Column(Integer)


class ExperimentRunStatus(enum.Enum):
    started = 1
    completed = 2
    failed = 3


class ExperimentRun(Base):

    __tablename__ = "experiment_runs"
    __table_args__ = {"schema": "triage_metadata"}

    run_id = Column("id", Integer, primary_key=True)
    start_time = Column(DateTime)
    start_method = Column(String)
    git_hash = Column(String)
    triage_version = Column(String)
    python_version = Column(String)
    experiment_hash = Column(
        String,
        ForeignKey("triage_metadata.experiments.experiment_hash")
    )
    platform = Column(Text)
    os_user = Column(Text)
    working_directory = Column(Text)
    ec2_instance_type = Column(Text)
    log_location = Column(Text)
    experiment_class_path = Column(Text)
    experiment_kwargs = Column(JSONB)
    installed_libraries = Column(ARRAY(Text))
    matrix_building_started = Column(DateTime)
    matrices_made = Column(Integer, default=0)
    matrices_skipped = Column(Integer, default=0)
    matrices_errored = Column(Integer, default=0)
    model_building_started = Column(DateTime)
    models_made = Column(Integer, default=0)
    models_skipped = Column(Integer, default=0)
    models_errored = Column(Integer, default=0)
    last_updated_time = Column(DateTime, onupdate=datetime.datetime.now)
    current_status = Column(Enum(ExperimentRunStatus))
    stacktrace = Column(Text)
    random_seed = Column(Integer)
    experiment_rel = relationship("Experiment")


class Subset(Base):

    __tablename__ = "subsets"
    __table_args__ = {"schema": "triage_metadata"}

    subset_hash = Column(String, primary_key=True)
    config = Column(JSONB)
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())


class ModelGroup(Base):

    __tablename__ = "model_groups"
    __table_args__ = {"schema": "triage_metadata"}

    model_group_id = Column(Integer, primary_key=True)
    model_type = Column(Text)
    hyperparameters = Column(JSONB)
    feature_list = Column(ARRAY(Text))
    model_config = Column(JSONB)


class ListPrediction(Base):

    __tablename__ = "list_predictions"
    __table_args__ = {"schema": "triage_metadata"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
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
    __table_args__ = {"schema": "triage_metadata"}

    experiment_hash = Column(
        String,
        ForeignKey("triage_metadata.experiments.experiment_hash"),
        primary_key=True
    )

    matrix_uuid = Column(String, primary_key=True)


class Matrix(Base):

    __tablename__ = "matrices"
    __table_args__ = {"schema": "triage_metadata"}

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
        String, ForeignKey("triage_metadata.experiments.experiment_hash")
    )
    feature_dictionary = Column(JSONB)


class Model(Base):

    __tablename__ = "models"
    __table_args__ = {"schema": "triage_metadata"}

    model_id = Column(Integer, primary_key=True)
    model_group_id = Column(
        Integer, ForeignKey("triage_metadata.model_groups.model_group_id")
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
        String, ForeignKey("triage_metadata.experiments.experiment_hash")
    )
    built_in_experiment_run = Column(
        Integer, ForeignKey("triage_metadata.experiment_runs.id")
    )
    train_end_time = Column(DateTime)
    test = Column(Boolean)
    train_matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    training_label_timespan = Column(Interval)
    model_size = Column(Float)
    random_seed = Column(Integer)

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
    __table_args__ = {"schema": "triage_metadata"}

    experiment_hash = Column(
        String,
        ForeignKey("triage_metadata.experiments.experiment_hash"),
        primary_key=True
    )
    model_hash = Column(String, primary_key=True)

    model_rel = relationship("Model", primaryjoin=(Model.model_hash == model_hash), foreign_keys=model_hash)
    experiment_rel = relationship("Experiment")


class FeatureImportance(Base):

    __tablename__ = "feature_importances"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
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
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric(6, 5))
    label_value = Column(Integer)
    rank_abs_no_ties = Column(Integer)
    rank_abs_with_ties = Column(Integer)
    rank_pct_no_ties = Column(Numeric(6, 5))
    rank_pct_with_ties = Column(Numeric(6, 5))
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    test_label_timespan = Column(Interval)

    model_rel = relationship("Model")
    matrix_rel = relationship("Matrix")


class TrainPrediction(Base):

    __tablename__ = "predictions"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    entity_id = Column(BigInteger, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    score = Column(Numeric(6, 5))
    label_value = Column(Integer)
    rank_abs_no_ties = Column(Integer)
    rank_abs_with_ties = Column(Integer)
    rank_pct_no_ties = Column(Float)
    rank_pct_with_ties = Column(Float)
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    test_label_timespan = Column(Interval)

    model_rel = relationship("Model")
    matrix_rel = relationship("Matrix")


class TestPredictionMetadata(Base):
    __tablename__ = "prediction_metadata"
    __table_args__ = {"schema": "test_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"), primary_key=True)
    tiebreaker_ordering = Column(Text)
    random_seed = Column(Integer)
    predictions_saved = Column(Boolean)


class TrainPredictionMetadata(Base):
    __tablename__ = "prediction_metadata"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"), primary_key=True)
    tiebreaker_ordering = Column(Text)
    random_seed = Column(Integer)
    predictions_saved = Column(Boolean)

class IndividualImportance(Base):

    __tablename__ = "individual_importances"
    __table_args__ = {"schema": "test_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
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
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    subset_hash = Column(String, primary_key=True, default='')
    evaluation_start_time = Column(DateTime, primary_key=True)
    evaluation_end_time = Column(DateTime, primary_key=True)
    as_of_date_frequency = Column(Interval, primary_key=True)
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    metric = Column(String, primary_key=True)
    parameter = Column(String, primary_key=True)
    num_labeled_examples = Column(Integer)
    num_labeled_above_threshold = Column(Integer)
    num_positive_labels = Column(Integer)
    sort_seed = Column(Integer)
    best_value = Column(Numeric)
    worst_value = Column(Numeric)
    stochastic_value = Column(Numeric)
    num_sort_trials = Column(Integer)
    standard_deviation = Column(Numeric)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")


class TrainEvaluation(Base):

    __tablename__ = "evaluations"
    __table_args__ = {"schema": "train_results"}

    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True
    )
    subset_hash = Column(String, primary_key=True, default='')
    evaluation_start_time = Column(DateTime, primary_key=True)
    evaluation_end_time = Column(DateTime, primary_key=True)
    as_of_date_frequency = Column(Interval, primary_key=True)
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    metric = Column(String, primary_key=True)
    parameter = Column(String, primary_key=True)
    num_labeled_examples = Column(Integer)
    num_labeled_above_threshold = Column(Integer)
    num_positive_labels = Column(Integer)
    sort_seed = Column(Integer)
    best_value = Column(Numeric)
    worst_value = Column(Numeric)
    stochastic_value = Column(Numeric)
    num_sort_trials = Column(Integer)
    standard_deviation = Column(Numeric)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")


class TestAequitas(Base):
    __tablename__ = "aequitas"
    __table_args__ = {"schema": "test_results"}
    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True, index=True
    )
    subset_hash = Column(String, primary_key=True, default='')
    tie_breaker = Column(String, primary_key=True)
    evaluation_start_time = Column(DateTime, primary_key=True, index=True)
    evaluation_end_time = Column(DateTime, primary_key=True, index=True)
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    parameter = Column(String, primary_key=True, index=True)
    attribute_name = Column(String, primary_key=True, index=True)
    attribute_value = Column(String, primary_key=True, index=True)
    total_entities = Column(Integer)
    group_label_pos = Column(Integer)
    group_label_neg = Column(Integer)
    group_size = Column(Integer)
    group_size_pct = Column(Numeric)
    prev = Column(Numeric)
    pp = Column(Integer)
    pn = Column(Integer)
    fp = Column(Integer)
    fn = Column(Integer)
    tn = Column(Integer)
    tp = Column(Integer)
    ppr = Column(Numeric)
    pprev = Column(Numeric)
    tpr = Column(Numeric)
    tnr = Column(Numeric)
    for_ = Column("for", Numeric)
    fdr = Column(Numeric)
    fpr = Column(Numeric)
    fnr = Column(Numeric)
    npv = Column(Numeric)
    precision = Column(Numeric)
    ppr_disparity = Column(Numeric)
    ppr_ref_group_value = Column(String)
    pprev_disparity = Column(Numeric)
    pprev_ref_group_value = Column(String)
    precision_disparity = Column(Numeric)
    precision_ref_group_value = Column(String)
    fdr_disparity = Column(Numeric)
    fdr_ref_group_value = Column(String)
    for_disparity = Column(Numeric)
    for_ref_group_value = Column(String)
    fpr_disparity = Column(Numeric)
    fpr_ref_group_value = Column(String)
    fnr_disparity = Column(Numeric)
    fnr_ref_group_value = Column(String)
    tpr_disparity = Column(Numeric)
    tpr_ref_group_value = Column(String)
    tnr_disparity = Column(Numeric)
    tnr_ref_group_value = Column(String)
    npv_disparity = Column(Numeric)
    npv_ref_group_value = Column(String)
    Statistical_Parity = Column(Boolean)
    Impact_Parity = Column(Boolean)
    FDR_Parity = Column(Boolean)
    FPR_Parity = Column(Boolean)
    FOR_Parity = Column(Boolean)
    FNR_Parity = Column(Boolean)
    TypeI_Parity = Column(Boolean)
    TypeII_Parity = Column(Boolean)
    Equalized_Odds = Column(Boolean)
    Unsupervised_Fairness = Column(Boolean)
    Supervised_Fairness = Column(Boolean)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")


class TrainAequitas(Base):
    __tablename__ = "aequitas"
    __table_args__ = {"schema": "train_results"}
    model_id = Column(
        Integer, ForeignKey("triage_metadata.models.model_id"), primary_key=True, index=True
    )
    subset_hash = Column(String, primary_key=True, default='')
    tie_breaker = Column(String, primary_key=True)
    evaluation_start_time = Column(DateTime, primary_key=True, index=True)
    evaluation_end_time = Column(DateTime, primary_key=True, index=True)
    matrix_uuid = Column(Text, ForeignKey("triage_metadata.matrices.matrix_uuid"))
    parameter = Column(String, primary_key=True, index=True)
    attribute_name = Column(String, primary_key=True, index=True)
    attribute_value = Column(String, primary_key=True, index=True)
    total_entities = Column(Integer)
    group_label_pos = Column(Integer)
    group_label_neg = Column(Integer)
    group_size = Column(Integer)
    group_size_pct = Column(Numeric)
    prev = Column(Numeric)
    pp = Column(Integer)
    pn = Column(Integer)
    fp = Column(Integer)
    fn = Column(Integer)
    tn = Column(Integer)
    tp = Column(Integer)
    ppr = Column(Numeric)
    pprev = Column(Numeric)
    tpr = Column(Numeric)
    tnr = Column(Numeric)
    for_ = Column("for", Numeric)
    fdr = Column(Numeric)
    fpr = Column(Numeric)
    fnr = Column(Numeric)
    npv = Column(Numeric)
    precision = Column(Numeric)
    ppr_disparity = Column(Numeric)
    ppr_ref_group_value = Column(String)
    pprev_disparity = Column(Numeric)
    pprev_ref_group_value = Column(String)
    precision_disparity = Column(Numeric)
    precision_ref_group_value = Column(String)
    fdr_disparity = Column(Numeric)
    fdr_ref_group_value = Column(String)
    for_disparity = Column(Numeric)
    for_ref_group_value = Column(String)
    fpr_disparity = Column(Numeric)
    fpr_ref_group_value = Column(String)
    fnr_disparity = Column(Numeric)
    fnr_ref_group_value = Column(String)
    tpr_disparity = Column(Numeric)
    tpr_ref_group_value = Column(String)
    tnr_disparity = Column(Numeric)
    tnr_ref_group_value = Column(String)
    npv_disparity = Column(Numeric)
    npv_ref_group_value = Column(String)
    Statistical_Parity = Column(Boolean)
    Impact_Parity = Column(Boolean)
    FDR_Parity = Column(Boolean)
    FPR_Parity = Column(Boolean)
    FOR_Parity = Column(Boolean)
    FNR_Parity = Column(Boolean)
    TypeI_Parity = Column(Boolean)
    TypeII_Parity = Column(Boolean)
    Equalized_Odds = Column(Boolean)
    Unsupervised_Fairness = Column(Boolean)
    Supervised_Fairness = Column(Boolean)

    matrix_rel = relationship("Matrix")
    model_rel = relationship("Model")


hash_partitioning_filename = os.path.join(
    os.path.dirname(__file__), "sql", "predictions_hash_partitioning.sql"
)
with open(hash_partitioning_filename) as fd:
    stmt = fd.read()

event.listen(Base.metadata, "after_create", DDL(stmt))
