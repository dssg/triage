import matplotlib
matplotlib.use('Agg')
import pytest
import testing.postgresql
import tempfile
from tests.utils import sample_config, populate_source_data
from triage import create_engine
from triage.component.catwalk.storage import ProjectStorage
from triage.component.catwalk.db import ensure_db
from tests.results_tests.factories import init_engine
from triage.component.postmodeling.crosstabs import CrosstabsConfigLoader
from triage.experiments import SingleThreadedExperiment


@pytest.fixture(name='db_engine', scope='function')
def fixture_db_engine():
    """pytest fixture provider to set up and teardown a "test" database
    and provide the test function a connection engine with which to
    query that database.

    """
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        yield engine
        engine.dispose()


@pytest.fixture(scope="function")
def db_engine_with_results_schema(db_engine):
    ensure_db(db_engine)
    init_engine(db_engine)
    yield db_engine


@pytest.fixture(scope="function")
def project_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def project_storage(project_path):
    """Set up a temporary project storage engine on the filesystem

    Yields (catwalk.storage.ProjectStorage)
    """
    yield ProjectStorage(project_path)


@pytest.fixture(scope='module')
def shared_db_engine():
    """pytest fixture provider to set up and teardown a "test" database
    and provide a test module a connection engine with which to
    query that database.

    """
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        yield engine
        engine.dispose()


@pytest.fixture(scope="module")
def shared_project_storage():
    """Set up a temporary project storage engine on the filesystem at module scope

    Yields (catwalk.storage.ProjectStorage)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_storage = ProjectStorage(temp_dir)
        yield project_storage


@pytest.fixture(scope="module")
def finished_experiment(shared_db_engine, shared_project_storage):
    """A successfully-run experiment. Its database schemas and project storage can be queried.

    Returns: (triage.experiments.SingleThreadedExperiment)
    """
    populate_source_data(shared_db_engine)
    base_config = sample_config()
    experiment = SingleThreadedExperiment(
        base_config,
        db_engine=shared_db_engine,
        project_path=shared_project_storage.project_path
    )
    experiment.run()
    return experiment


@pytest.fixture(scope="module")
def finished_experiment_without_predictions(shared_db_engine, shared_project_storage):
    """A successfully-run experiment. Its database schemas and project storage can be queried.

    Returns: (triage.experiments.SingleThreadedExperiment)
    """
    populate_source_data(shared_db_engine)
    base_config = sample_config()
    experiment = SingleThreadedExperiment(
        base_config,
        db_engine=shared_db_engine,
        project_path=shared_project_storage.project_path,
        save_predictions=False
    )
    experiment.run()
    return experiment


@pytest.fixture(scope='module')
def crosstabs_config():
    """Example crosstabs config.

    Should work with after an experiment run with tests.utils.sample_config
    """
    return CrosstabsConfigLoader(config={
        "output": {
          "schema": 'test_results',
          "table": 'crosstabs'
        },
        "thresholds": {
            "rank_abs": [50],
            "rank_pct": [],
        },
        "entity_id_list": [],
        "models_list_query": "select unnest(ARRAY[1]) :: int as model_id",
        "as_of_dates_query": "select unnest(ARRAY['2012-06-01']) :: date as as_of_date",
        "models_dates_join_query": """
    select model_id,
          as_of_date
          from models_list_query m
          cross join as_of_dates_query a join (select distinct model_id, as_of_date from test_results.predictions) p
          using (model_id, as_of_date)""",
        "features_query": """
select m.model_id, f1.*
 from features.entity_features_aggregation_imputed f1 join
 models_dates_join_query m using (as_of_date)""",
        "predictions_query": """
select model_id,
      as_of_date,
      entity_id,
      score,
      label_value,
      coalesce(rank_abs_no_ties, row_number() over (partition by (model_id, as_of_date) order by score desc)) as rank_abs,
      coalesce(rank_pct_no_ties*100, ntile(100) over (partition by (model_id, as_of_date) order by score desc)) as rank_pct
  from test_results.predictions
  JOIN models_dates_join_query USING(model_id, as_of_date)
  where model_id IN (select model_id from models_list_query)
  AND as_of_date in (select as_of_date from as_of_dates_query)""",
    })


@pytest.fixture(scope="module")
def sample_timechop_splits():
    return [
        {
            "feature_start_time": "2010-01-01T00:00:00",
            "feature_end_time": "2014-01-01T00:00:00",
            "label_start_time": "2011-01-01T00:00:00",
            "label_end_time": "2014-01-01T00:00:00",
            "train_matrix": {
                "first_as_of_time": "2011-06-01T00:00:00",
                "last_as_of_time": "2011-12-01T00:00:00",
                "matrix_info_end_time": "2012-06-01T00:00:00",
                "as_of_times": [
                    "2011-06-01T00:00:00",
                    "2011-07-01T00:00:00",
                    "2011-08-01T00:00:00",
                    "2011-09-01T00:00:00",
                    "2011-10-01T00:00:00",
                    "2011-11-01T00:00:00",
                    "2011-12-01T00:00:00"
                ],
                "training_label_timespan": "6months",
                "training_as_of_date_frequency": "1month",
                "max_training_history": "6months"
            },
            "test_matrices": [
                {
                    "first_as_of_time": "2012-06-01T00:00:00",
                    "last_as_of_time": "2012-06-01T00:00:00",
                    "matrix_info_end_time": "2012-12-01T00:00:00",
                    "as_of_times": [
                        "2012-06-01T00:00:00"
                    ],
                    "test_label_timespan": "6months",
                    "test_as_of_date_frequency": "3months",
                    "test_duration": "1months"
                }
            ],
            "train_uuid": "40de3a41a7b210c6a525adeb74fafb22",
            "test_uuids": [
                "6c41a75c5270ed036370ca2344371150"
            ]
        },
        {
            "feature_start_time": "2010-01-01T00:00:00",
            "feature_end_time": "2014-01-01T00:00:00",
            "label_start_time": "2011-01-01T00:00:00",
            "label_end_time": "2014-01-01T00:00:00",
            "train_matrix": {
                "first_as_of_time": "2012-06-01T00:00:00",
                "last_as_of_time": "2012-12-01T00:00:00",
                "matrix_info_end_time": "2013-06-01T00:00:00",
                "as_of_times": [
                    "2012-06-01T00:00:00",
                    "2012-07-01T00:00:00",
                    "2012-08-01T00:00:00",
                    "2012-09-01T00:00:00",
                    "2012-10-01T00:00:00",
                    "2012-11-01T00:00:00",
                    "2012-12-01T00:00:00"
                ],
                "training_label_timespan": "6months",
                "training_as_of_date_frequency": "1month",
                "max_training_history": "6months"
            },
            "test_matrices": [
                {
                    "first_as_of_time": "2013-06-01T00:00:00",
                    "last_as_of_time": "2013-06-01T00:00:00",
                    "matrix_info_end_time": "2013-12-01T00:00:00",
                    "as_of_times": [
                        "2013-06-01T00:00:00"
                    ],
                    "test_label_timespan": "6months",
                    "test_as_of_date_frequency": "3months",
                    "test_duration": "1months"
                }
            ],
            "train_uuid": "95f998f70d5be1cf3d2ec833cd9db079",
            "test_uuids": [
                "8fd8be5c0b8b2e5b06a233b960769ccf"
            ]
        }
    ]


@pytest.fixture(scope="module")
def sample_grid_config():
    return {
        'sklearn.tree.DecisionTreeClassifier': {
            'max_depth': [2,10],
            'min_samples_split': [2],
        },
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_jobs': [-1],
            'n_estimators': [10],
            'criterion': ['gini'],
            'max_depth': [1],
            'max_features': ['sqrt'],
            'min_samples_split': [2,5]
        },
        'sklearn.ensemble.GradientBoostingClassifier': {
            'loss': ['deviance', 'exponential']
        },
        'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 1]
        },
        'triage.component.catwalk.baselines.rankers.PercentileRankOneFeature': {
            'feature': ['feature_one', 'feature_two'],
            'descend': [True]
        },
        'triage.component.catwalk.baselines.thresholders.SimpleThresholder': {
            'rules': [['feature_one > 3', 'feature_two <= 5']],
            'logical_operator': ['and']
        }
    }
