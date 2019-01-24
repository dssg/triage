import pytest
import testing.postgresql
import tempfile
from sqlalchemy import create_engine
from triage.component.catwalk.storage import ProjectStorage
from triage.component.catwalk.db import ensure_db
from tests.results_tests.factories import init_engine


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
