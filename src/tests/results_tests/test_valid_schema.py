from triage.component.results_schema import Base


def test_full_schema(db_engine):
    Base.metadata.create_all(bind=db_engine)
