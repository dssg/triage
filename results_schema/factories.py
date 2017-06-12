"""Testing Factories for creating database objects in unit tests

If init_engine is called first, objects are instantiated in the session at
module level. Example:

```
from results_schema.factories import EvaluationFactory, session, init_engine

engine = # your engine creation code here
init_engine(engine)
EvaluationFactory()
session.commit()

results = engine.execute('select * from results.evaluations')
```
"""

import factory 
import factory.fuzzy
from results_schema import schema
import uuid
from datetime import datetime
import testing.postgresql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, create_session

sessionmaker = sessionmaker()
session = scoped_session(sessionmaker)

class ExperimentFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Experiment
        sqlalchemy_session = session
    experiment_hash = factory.fuzzy.FuzzyText()
    config = {}

class ModelGroupFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.ModelGroup
        sqlalchemy_session = session
    model_type = 'sklearn.ensemble.RandomForestClassifier'
    model_parameters = {'hyperparam1': 'value1', 'hyperparam2': 'value2'}
    feature_list = ['feature1', 'feature2', 'feature3']
    model_config = {}

class ModelFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Model
        sqlalchemy_session = session
    model_group_rel = factory.SubFactory(ModelGroupFactory)
    model_hash = factory.fuzzy.FuzzyText()
    run_time = factory.LazyFunction(lambda: datetime.now())
    batch_run_time = factory.LazyFunction(lambda: datetime.now())
    model_type = 'sklearn.ensemble.RandomForestClassifier'
    model_parameters = {'hyperparam1': 'value1', 'hyperparam2': 'value2'}
    model_comment = ''
    batch_comment = ''
    config = {}
    experiment_rel = factory.SubFactory(ExperimentFactory)
    train_end_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    test = False
    train_matrix_uuid = factory.fuzzy.FuzzyText()
    train_label_window = '1y'

class FeatureImportanceFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.FeatureImportance
        sqlalchemy_session = session
    model_rel = factory.SubFactory(ModelFactory)
    feature = factory.fuzzy.FuzzyText()
    feature_importance = factory.fuzzy.FuzzyDecimal(0, 1)
    rank_abs = 1
    rank_pct = 1.0

class PredictionFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Prediction
        sqlalchemy_session = session

    model_rel = factory.SubFactory(ModelFactory)
    entity_id = factory.fuzzy.FuzzyInteger(0)
    as_of_date = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    score = factory.fuzzy.FuzzyDecimal(0, 1)
    label_value = factory.fuzzy.FuzzyInteger(0, 1)
    rank_abs = 1
    rank_pct = 1.0
    matrix_uuid = factory.fuzzy.FuzzyText()
    test_label_window = '3m'


class EvaluationFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = schema.Evaluation
        sqlalchemy_session = session
    model_rel = factory.SubFactory(ModelFactory)
    evaluation_start_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    evaluation_end_time = factory.fuzzy.FuzzyNaiveDateTime(datetime(2008, 1, 1))
    example_frequency = '3d'
    metric = 'precision@'
    parameter = '100_abs'
    value = factory.fuzzy.FuzzyDecimal(0, 1)
    num_labeled_examples = 10
    num_labeled_above_threshold = 8
    num_positive_labels = 5
    sort_seed = 8

def init_engine(new_engine):
    global sessionmaker, engine, session
    engine = new_engine
    session.remove()
    sessionmaker.configure(bind=engine)
