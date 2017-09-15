from catwalk.db import ensure_db
from catwalk.individual_importance import IndividualImportanceCalculator
from catwalk.storage import InMemoryModelStorageEngine
from tests.utils import fake_trained_model, sample_metta_csv_diff_order

import tempfile
from sqlalchemy import create_engine
import testing.postgresql
from unittest.mock import patch


def sample_individual_importance_strategy(
    db_engine,
    model_id,
    as_of_date,
    test_matrix_store,
    n_ranks
):
    return [{
        'entity_id': 1,
        'feature_value': 0.5,
        'feature_name': 'm_feature',
        'score': 0.5,
    }, {
        'entity_id': 1,
        'feature_value': 0.5,
        'feature_name': 'k_feature',
        'score': 0.5,
    }]


@patch.dict(
    'catwalk.individual_importance.CALCULATE_STRATEGIES',
    {'sample': sample_individual_importance_strategy}
)
def test_calculate_and_save():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        project_path = 'econ-dev/inspections'
        with tempfile.TemporaryDirectory() as temp_dir:
            train_store, test_store = sample_metta_csv_diff_order(temp_dir)
            model_storage_engine = InMemoryModelStorageEngine(project_path)
            calculator = IndividualImportanceCalculator(db_engine, methods=['sample'])
            # given a trained model
            # and a test matrix
            _, model_id = \
                fake_trained_model(
                    project_path,
                    model_storage_engine,
                    db_engine,
                    train_matrix_uuid=train_store.uuid
                )
            # i expect to be able to call calculate and save
            calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
            # and find individual importances in the results schema afterwards
            records = [
                row for row in
                db_engine.execute('''select entity_id, as_of_date
                from results.individual_importances
                join results.models using (model_id)''')
            ]
            assert len(records) > 0
            # and that when run again, has the same result
            calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
            new_records = [
                row for row in
                db_engine.execute('''select entity_id, as_of_date
                from results.individual_importances
                join results.models using (model_id)''')
            ]
            assert len(records) == len(new_records)
            assert records == new_records
