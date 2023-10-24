from datetime import datetime, timedelta
import testing.postgresql
from sqlalchemy.engine import create_engine

from . import utils

from triage.component.architect.plotting import inspect_cohort_query_on_date, CohortInspectionResults

def test_inspect_cohort_query_on_date():
    input_data = [
        (1, datetime(2016, 1, 1), True),
        (1, datetime(2016, 4, 1), False),
        (1, datetime(2016, 3, 1), True),
        (2, datetime(2016, 1, 1), False),
        (2, datetime(2016, 1, 1), True),
        (3, datetime(2016, 1, 1), True),
        (5, datetime(2016, 3, 1), True),
        (5, datetime(2016, 4, 1), True),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        utils.create_binary_outcome_events(engine, "events", input_data)
        results = inspect_cohort_query_on_date(
            db_engine=engine,
            query="select entity_id from events where outcome_date < '{as_of_date}'::date",
            as_of_date=datetime(2016, 2, 1)
        )

        expected_output = CohortInspectionResults(
            ran_successfully=True,
            num_rows=3,
            num_distinct_entity_ids=3,
            examples=[1, 2, 3]
        )
            
        assert results == expected_output

def test_inspect_cohort_query_on_date_unsuccessful():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        results = inspect_cohort_query_on_date(
            db_engine=engine,
            query="select entity_id from events2 where outcome_date < '{as_of_date}'::date",
            as_of_date=datetime(2016, 2, 1)
        )

        expected_output = CohortInspectionResults(
            ran_successfully=False,
            num_rows=0,
            num_distinct_entity_ids=0,
            examples=[]
        )
            
        assert results == expected_output
