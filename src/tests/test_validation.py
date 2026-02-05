from unittest import mock

from tests.utils import open_side_effect, populate_source_data, sample_config
from triage.component.catwalk.db import ensure_db
from triage.experiments.validate import ExperimentValidator


def test_experiment_validator(db_engine):
    ensure_db(db_engine)
    populate_source_data(db_engine)
    with mock.patch("triage.util.conf.open", side_effect=open_side_effect) as mock_file:
        ExperimentValidator(db_engine).run(sample_config("query"))
        ExperimentValidator(db_engine).run(sample_config("filepath"))
