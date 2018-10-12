from triage.component.catwalk.model_trainers import ModelTrainer
from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.evaluation import ModelEvaluator
from triage.component.catwalk.utils import save_experiment_and_get_hash
from triage.component.catwalk.storage import (
    ModelStorageEngine,
)
from tests.utils import (
    rig_engines,
    get_matrix_store,
    matrix_creator,
    matrix_metadata_creator,
)

import datetime
import pandas


def test_integration():
    with rig_engines() as (db_engine, project_storage):
        train_store = get_matrix_store(
            project_storage,
            matrix_creator(),
            matrix_metadata_creator(matrix_type="train"),
        )
        as_of_dates = [datetime.date(2016, 12, 21), datetime.date(2017, 1, 21)]

        test_stores = []
        for as_of_date in as_of_dates:
            matrix_store = get_matrix_store(
                project_storage,
                pandas.DataFrame.from_dict(
                    {
                        "entity_id": [3],
                        "feature_one": [8],
                        "feature_two": [5],
                        "label": [0],
                    }
                ).set_index("entity_id"),
                matrix_metadata_creator(end_time=as_of_date, indices=["entity_id"]),
            )
            test_stores.append(matrix_store)

        model_storage_engine = ModelStorageEngine(project_storage)

        experiment_hash = save_experiment_and_get_hash({}, db_engine)
        # instantiate pipeline objects
        trainer = ModelTrainer(
            experiment_hash=experiment_hash,
            model_storage_engine=model_storage_engine,
            db_engine=db_engine,
        )
        predictor = Predictor(model_storage_engine, db_engine)
        model_evaluator = ModelEvaluator(
            [{"metrics": ["precision@"], "thresholds": {"top_n": [5]}}], [{}], db_engine
        )

        # run the pipeline
        grid_config = {
            "sklearn.linear_model.LogisticRegression": {
                "C": [0.00001, 0.0001],
                "penalty": ["l1", "l2"],
                "random_state": [2193],
            }
        }
        model_ids = trainer.train_models(
            grid_config=grid_config, misc_db_parameters=dict(), matrix_store=train_store
        )

        for model_id in model_ids:
            for as_of_date, test_store in zip(as_of_dates, test_stores):
                predictions_proba = predictor.predict(
                    model_id,
                    test_store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=["feature_one", "feature_two"],
                )

                model_evaluator.evaluate(predictions_proba, test_store, model_id)

        # assert
        # 1. that the predictions table entries are present and
        # can be linked to the original models
        records = [
            row
            for row in db_engine.execute(
                """select entity_id, model_id, as_of_date
            from test_results.predictions
            join model_metadata.models using (model_id)
            order by 3, 2"""
            )
        ]
        assert records == [
            (3, 1, datetime.datetime(2016, 12, 21)),
            (3, 2, datetime.datetime(2016, 12, 21)),
            (3, 3, datetime.datetime(2016, 12, 21)),
            (3, 4, datetime.datetime(2016, 12, 21)),
            (3, 1, datetime.datetime(2017, 1, 21)),
            (3, 2, datetime.datetime(2017, 1, 21)),
            (3, 3, datetime.datetime(2017, 1, 21)),
            (3, 4, datetime.datetime(2017, 1, 21)),
        ]

        # that evaluations are there
        records = [
            row
            for row in db_engine.execute(
                """
                select model_id, evaluation_start_time, metric, parameter
                from test_results.evaluations order by 2, 1"""
            )
        ]
        assert records == [
            (1, datetime.datetime(2016, 12, 21), "precision@", "5_abs"),
            (2, datetime.datetime(2016, 12, 21), "precision@", "5_abs"),
            (3, datetime.datetime(2016, 12, 21), "precision@", "5_abs"),
            (4, datetime.datetime(2016, 12, 21), "precision@", "5_abs"),
            (1, datetime.datetime(2017, 1, 21), "precision@", "5_abs"),
            (2, datetime.datetime(2017, 1, 21), "precision@", "5_abs"),
            (3, datetime.datetime(2017, 1, 21), "precision@", "5_abs"),
            (4, datetime.datetime(2017, 1, 21), "precision@", "5_abs"),
        ]
