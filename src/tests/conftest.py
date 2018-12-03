import pytest
import testing.postgresql
from sqlalchemy import create_engine


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
