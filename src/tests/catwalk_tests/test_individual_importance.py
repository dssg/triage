from triage.component.catwalk.individual_importance import (
    IndividualImportanceCalculator,
    IndividualImportanceCalculatorNoOp,
)
from tests.utils import (
    rig_engines,
    fake_trained_model,
    matrix_creator,
    matrix_metadata_creator,
    get_matrix_store,
)

from unittest.mock import patch


def sample_individual_importance_strategy(
    db_engine, model_id, as_of_date, test_matrix_store, n_ranks
):
    return [
        {
            "entity_id": 1,
            "feature_value": 0.5,
            "feature_name": "m_feature",
            "score": 0.5,
        },
        {
            "entity_id": 1,
            "feature_value": 0.5,
            "feature_name": "k_feature",
            "score": 0.5,
        },
    ]


@patch.dict(
    "triage.component.catwalk.individual_importance.CALCULATE_STRATEGIES",
    {"sample": sample_individual_importance_strategy},
)
def test_calculate_and_save():
    with rig_engines() as (db_engine, project_storage):
        train_store = get_matrix_store(
            project_storage,
            matrix_creator(),
            matrix_metadata_creator(matrix_type="train"),
        )
        test_store = get_matrix_store(
            project_storage,
            matrix_creator(),
            matrix_metadata_creator(matrix_type="test"),
        )
        calculator = IndividualImportanceCalculator(
            db_engine, methods=["sample"], replace=False
        )
        # given a trained model
        # and a test matrix
        _, model_id = fake_trained_model(db_engine, train_matrix_uuid=train_store.uuid)
        # i expect to be able to call calculate and save
        calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
        # and find individual importances in the results schema afterwards
        records = [
            row
            for row in db_engine.execute(
                """select entity_id, as_of_date
            from test_results.individual_importances
            join triage_metadata.models using (model_id)"""
            )
        ]
        assert len(records) > 0
        # and that when run again, has the same result
        calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
        new_records = [
            row
            for row in db_engine.execute(
                """select entity_id, as_of_date
            from test_results.individual_importances
            join triage_metadata.models using (model_id)"""
            )
        ]
        assert len(records) == len(new_records)
        assert records == new_records


@patch.dict(
    "triage.component.catwalk.individual_importance.CALCULATE_STRATEGIES",
    {"sample": sample_individual_importance_strategy},
)
def test_calculate_and_save_noop():
    with rig_engines() as (db_engine, project_storage):
        train_store = get_matrix_store(
            project_storage,
            matrix_creator(),
            matrix_metadata_creator(matrix_type="train"),
        )
        test_store = get_matrix_store(
            project_storage,
            matrix_creator(),
            matrix_metadata_creator(matrix_type="test"),
        )
        calculator = IndividualImportanceCalculatorNoOp()
        # given a trained model
        # and a test matrix
        _, model_id = fake_trained_model(db_engine, train_matrix_uuid=train_store.uuid)
        # i expect to be able to call calculate and save
        calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
        # and find individual importances in the results schema afterwards
        records = [
            row
            for row in db_engine.execute(
                """select entity_id, as_of_date
            from test_results.individual_importances
            join triage_metadata.models using (model_id)"""
            )
        ]
        assert len(records) == 0
        # and that when run again, has the same result
        calculator.calculate_and_save_all_methods_and_dates(model_id, test_store)
        new_records = [
            row
            for row in db_engine.execute(
                """select entity_id, as_of_date
            from test_results.individual_importances
            join triage_metadata.models using (model_id)"""
            )
        ]
        assert len(records) == len(new_records)
        assert records == new_records
