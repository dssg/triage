"""Testing Factories for creating database objects in unit tests

Factories now require an explicit session to be passed in via _session parameter.
Example:

```
from .factories import EvaluationFactory

# In your test
def test_something(db_session):
    evaluation = EvaluationFactory(_session=db_session)
    db_session.commit()
```
"""

from datetime import datetime

import factory
import factory.fuzzy
from sqlalchemy.orm import scoped_session, sessionmaker

from triage.component import results_schema as schema

ScopedSession = scoped_session(sessionmaker())
# Alias for backward compatibility with existing test imports
session = ScopedSession


def init_engine(engine):
    """Initialize the factories with a database engine.

    This configures the scoped session to use the provided engine.
    Should be called after ensure_db() sets up the schema.

    :param engine: SQLAlchemy engine instance
    """
    ScopedSession.remove()
    ScopedSession.configure(bind=engine)


def set_session(session):
    """
    This should be called at the start of each test

    :param session: SQLAlchemy session to use for this test/thread
    """
    ScopedSession.remove()
    ScopedSession.configure(bind=session.bind)


def clear_session():
    ScopedSession.remove()


class ExperimentFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Experiment
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    experiment_hash = factory.fuzzy.FuzzyText()
    config = {}


class ModelGroupFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.ModelGroup
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_type = "sklearn.ensemble.RandomForestClassifier"
    hyperparameters = {"hyperparam1": "value1", "hyperparam2": "value2"}
    feature_list = ["feature1", "feature2", "feature3"]
    model_config = {}


class MatrixFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Matrix
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    matrix_uuid = factory.fuzzy.FuzzyText()


class BaseModelFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Model
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_group_rel = factory.SubFactory(ModelGroupFactory)
    model_hash = factory.fuzzy.FuzzyText()
    run_time = factory.LazyFunction(lambda: datetime.now())
    batch_run_time = factory.LazyFunction(lambda: datetime.now())
    model_type = "sklearn.ensemble.RandomForestClassifier"
    hyperparameters = {"hyperparam1": "value1", "hyperparam2": "value2"}
    model_comment = ""
    batch_comment = ""
    config = {}
    matrix_rel = factory.SubFactory(MatrixFactory)
    train_end_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    test = False
    train_matrix_uuid = factory.SelfAttribute("matrix_rel.matrix_uuid")
    training_label_timespan = "1y"


class ExperimentModelFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.ExperimentModel
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_rel = factory.SubFactory(BaseModelFactory)
    experiment_rel = factory.SubFactory(ExperimentFactory)


class ModelFactory(BaseModelFactory):
    experiment_association = factory.RelatedFactory(ExperimentModelFactory, "model_rel")


class SubsetFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Subset
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    subset_hash = factory.fuzzy.FuzzyText()
    config = {}
    created_timestamp = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))


class FeatureImportanceFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.FeatureImportance
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    # model_rel = factory.SubFactory(ModelFactory)
    feature = factory.fuzzy.FuzzyText()
    feature_importance = factory.fuzzy.FuzzyDecimal(0, 1)
    rank_abs = 1
    rank_pct = 1.0


class PredictionFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.TestPrediction
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_rel = factory.SubFactory(ModelFactory)
    entity_id = factory.fuzzy.FuzzyInteger(0)
    as_of_date = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    score = factory.fuzzy.FuzzyDecimal(0, 1)
    label_value = factory.fuzzy.FuzzyInteger(0, 1)
    matrix_rel = factory.SubFactory(MatrixFactory)
    test_label_timespan = "3m"


class ListPredictionFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.ListPrediction
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_rel = factory.SubFactory(ModelFactory)
    entity_id = factory.fuzzy.FuzzyInteger(0)
    as_of_date = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    score = factory.fuzzy.FuzzyDecimal(0, 1)
    rank_abs = 1
    rank_pct = 1.0
    matrix_uuid = "efgh"
    test_label_timespan = "3m"


class IndividualImportanceFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.IndividualImportance
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_rel = factory.SubFactory(ModelFactory)
    entity_id = factory.fuzzy.FuzzyInteger(0)
    as_of_date = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    feature = factory.fuzzy.FuzzyText()
    feature_value = factory.fuzzy.FuzzyDecimal(0, 100)
    method = factory.fuzzy.FuzzyText()
    importance_score = factory.fuzzy.FuzzyDecimal(0, 1)


class EvaluationFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.TestEvaluation
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    model_rel = factory.SubFactory(ModelFactory)
    subset_hash = ""
    evaluation_start_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    evaluation_end_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    as_of_date_frequency = "3d"
    metric = "precision@"
    parameter = "100_abs"
    num_labeled_examples = 10
    num_labeled_above_threshold = 8
    num_positive_labels = 5
    sort_seed = 8
    best_value = factory.fuzzy.FuzzyDecimal(0, 1)
    worst_value = factory.fuzzy.FuzzyDecimal(0, 1)
    stochastic_value = factory.fuzzy.FuzzyDecimal(0, 1)
    num_sort_trials = 5
    standard_deviation = 0.05
    matrix_rel = factory.SubFactory(MatrixFactory)
    matrix_uuid = factory.SelfAttribute("matrix_rel.matrix_uuid")


class TriageRunFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.TriageRun
        sqlalchemy_session = ScopedSession
        sqlalchemy_session_persistence = "commit"
        # sqlalchemy_session_persistence = "flush"

    # experiment_rel = factory.SubFactory(ExperimentFactory)

    start_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    start_method = "run"
    git_hash = "abcd"
    triage_version = "4.2.0"
    python_version = "3.6.2 (default, May 28 2020, 13:23:43) \n[GCC 9.3.0]"
    platform = "Linux!!!"
    os_user = "dsapp"
    working_directory = "/the/best/directory"
    ec2_instance_type = "x2.128xlarge"
    log_location = "/the/logs"
    experiment_class_path = "triage.experiments.singlethreaded.SingleThreadedExperiment"
    experiment_kwargs = {}
    installed_libraries = ["triage"]
    matrix_building_started = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    matrices_made = 0
    matrices_skipped = 0
    matrices_errored = 0
    model_building_started = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    models_made = 0
    models_skipped = 0
    models_errored = 0
    last_updated_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    current_status = schema.TriageRunStatus.started
    stacktrace = ""
