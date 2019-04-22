from sqlalchemy import create_engine
import testing.postgresql

from triage.component.catwalk.db import ensure_db

from tests.utils import sample_config, populate_source_data
from triage.experiments.validate import ExperimentValidator


def test_experiment_validator():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        ExperimentValidator(db_engine).run(sample_config())
