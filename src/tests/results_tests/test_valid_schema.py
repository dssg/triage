import testing.postgresql
from sqlalchemy import create_engine

from triage.component.results_schema import Base


def test_full_schema():
    with testing.postgresql.Postgresql() as postgres:
        engine = create_engine(postgres.url())
        Base.metadata.create_all(bind=engine)
