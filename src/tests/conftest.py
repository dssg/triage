import pytest
import testing.postgresql
import tempfile
from tests.utils import sample_config, populate_source_data
from sqlalchemy import create_engine
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
def project_storage():
    """Set up a temporary project storage engine on the filesystem

    Yields (catwalk.storage.ProjectStorage)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_storage = ProjectStorage(temp_dir)
        yield project_storage


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
      coalesce(rank_abs, row_number() over (partition by (model_id, as_of_date) order by score desc)) as rank_abs,
      coalesce(rank_pct*100, ntile(100) over (partition by (model_id, as_of_date) order by score desc)) as rank_pct
  from test_results.predictions
  JOIN models_dates_join_query USING(model_id, as_of_date)
  where model_id IN (select model_id from models_list_query)
  AND as_of_date in (select as_of_date from as_of_dates_query)""",
    })
