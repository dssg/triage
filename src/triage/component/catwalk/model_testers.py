import logging

from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.individual_importance import (
    IndividualImportanceCalculator,
)
from triage.component.catwalk.evaluation import ModelEvaluator


class ModelTester(object):
    def __init__(
        self,
        db_engine,
        model_storage_engine,
        matrix_storage_engine,
        replace,
        save_predictions,
        evaluator_config,
        individual_importance_config,
    ):
        self.matrix_storage_engine = matrix_storage_engine
        self.replace = replace
        self.predictor = Predictor(
            db_engine=db_engine,
            model_storage_engine=model_storage_engine,
            replace=replace,
            save_predictions=save_predictions,
        )

        self.individual_importance_calculator = IndividualImportanceCalculator(
            db_engine=db_engine,
            n_ranks=individual_importance_config.get("n_ranks", 5),
            methods=individual_importance_config.get("methods", ["uniform"]),
            replace=replace,
        )

        self.evaluator = ModelEvaluator(
            db_engine=db_engine,
            sort_seed=evaluator_config.get("sort_seed", None),
            testing_metric_groups=evaluator_config.get("testing_metric_groups", []),
            training_metric_groups=evaluator_config.get("training_metric_groups", []),
        )

    def generate_model_test_tasks(self, split, train_store, model_ids):
        test_tasks = []
        for test_matrix_def, test_uuid in zip(
            split["test_matrices"], split["test_uuids"]
        ):
            test_store = self.matrix_storage_engine.get_store(test_uuid)

            if test_store.empty:
                logging.warning(
                    """Test matrix for uuid %s
                was empty, no point in generating predictions. Not creating test task.
                """,
                    test_uuid,
                )
                continue
            test_tasks.append(
                {
                    "test_store": test_store,
                    "train_store": train_store,
                    "model_ids": [model_id for model_id in model_ids if model_id],
                }
            )
        return test_tasks

    def process_model_test_task(self, test_store, train_store, model_ids):
        as_of_times = test_store.metadata["as_of_times"]
        logging.info(
            "Testing and scoring all model ids with test matrix %s. "
            "as_of_times min: %s max: %s num: %s",
            test_store.uuid,
            min(as_of_times),
            max(as_of_times),
            len(as_of_times),
        )

        for model_id in model_ids:
            logging.info("Testing model id %s", model_id)

            self.individual_importance_calculator.calculate_and_save_all_methods_and_dates(
                model_id, test_store
            )

            # Generate predictions for the testing data then training data
            for store in (test_store, train_store):
                if self.replace:
                    should_run = True
                    reason = "Replace flag was set"
                elif self.evaluator.needs_evaluations(store, model_id):
                    should_run = True
                    reason = "Needed evaluations are not all present in DB"
                elif self.predictor.needs_predictions(store, model_id):
                    should_run = True
                    reason = "Needed predictions are not all present in DB"
                else:
                    should_run = False
                    reason = None

                if should_run:
                    logging.info(
                        "Predicting and evaluating for matrix %s-%s and model %s . Reason? %s",
                        store.uuid,
                        store.matrix_type,
                        model_id,
                        reason
                    )
                    predictions_proba = self.predictor.predict(
                        model_id,
                        store,
                        misc_db_parameters=dict(),
                        train_matrix_columns=train_store.columns(),
                    )

                    self.evaluator.evaluate(
                        predictions_proba=predictions_proba,
                        matrix_store=store,
                        model_id=model_id,
                    )
                else:
                    logging.info(
                        "No need to predict/evaluate for matrix %s-%s and model %s. Skipping",
                        store.uuid,
                        store.matrix_type,
                        model_id
                    )
