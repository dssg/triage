from triage.util.db import run_statements
import pytest
import sqlalchemy
from sqlalchemy import text as t


def test_run_statements(db_engine):
    """Test that database connections are cleaned up regardless of in-transaction
    query errors.
    """
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        run_statements(['insert into blah'], db_engine)

    ((query_count,),) = db_engine.execute(
        t("""\
            select count(1) from pg_stat_activity
            where datname = :datname and
                  query not ilike '%%pg_stat_activity%%'
        """),
        datname=db_engine.url.database,
    )

    assert query_count == 0
