"""Main application"""
from .model_trainers import ModelTrainer
from .predictors import Predictor
from .evaluation import ModelEvaluator
from .individual_importance import IndividualImportanceCalculator
from .model_grouping import ModelGrouper

import logging


class ModelTrainTester(object):
    def __init__(
        self,
        matrix_storage_engine,
        model_trainer,
        model_evaluator,
        individual_importance_calculator,
        predictor
    ):
        self.matrix_storage_engine = matrix_storage_engine
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.individual_importance_calculator = individual_importance_calculator
        self.predictor = predictor

    def generate_tasks(self, split, grid_config, model_comment=None):
        logging.info("Generating train/test tasks for split %s", split["train_uuid"])
        train_store = self.matrix_storage_engine.get_store(split["train_uuid"])
        if train_store.empty:
            logging.warning(
                """Train matrix for split %s was empty,
            no point in training this model. Skipping
            """,
                split["train_uuid"],
            )
            return []
        if len(train_store.labels.unique()) == 1:
            logging.warning(
                """Train Matrix for split %s had only one
            unique value, no point in training this model. Skipping
            """,
                split["train_uuid"],
            )
            return []
        train_tasks = self.model_trainer.generate_train_tasks(
            grid_config=grid_config,
            misc_db_parameters=dict(test=False, model_comment=model_comment),
            matrix_store=train_store
        )

        train_test_tasks = []
        for test_matrix_def, test_uuid in zip(
            split["test_matrices"], split["test_uuids"]
        ):
            test_store = self.matrix_storage_engine.get_store(test_uuid)

            if test_store.empty:
                logging.warning(
                    """Test matrix for uuid %s
                was empty, no point in generating predictions. Not creating train/test task.
                """,
                    test_uuid,
                )
                continue
            for train_task in train_tasks:
                train_test_tasks.append(
                    {
                        "test_store": test_store,
                        "train_store": train_store,
                        "train_kwargs": train_task,
                    }
                )
        return train_test_tasks

    def process_all_tasks(self, tasks):
        for task in tasks:
            self.process_task(**task)

    def process_task(self, test_store, train_store, train_kwargs):
        logging.info("Beginning train task %s", train_kwargs)
        with self.model_trainer.cache_models(), test_store.cache(), train_store.cache():
            # will cache any trained models until it goes out of scope (at the end of the task)
            # this way we avoid loading the model pickle again for predictions
            model_id = self.model_trainer.process_train_task(**train_kwargs)
            if not model_id:
                logging.warning("No model id returned from ModelTrainer.process_train_task, "
                                "training unsuccessful. Not attempting to test")
                return
            logging.info("Trained task %s and got model id %s", train_kwargs, model_id)
            as_of_times = test_store.metadata["as_of_times"]
            logging.info(
                "Testing and scoring model id %s with test matrix %s. "
                "as_of_times min: %s max: %s num: %s",
                model_id,
                test_store.uuid,
                min(as_of_times),
                max(as_of_times),
                len(as_of_times),
            )

            self.individual_importance_calculator.calculate_and_save_all_methods_and_dates(
                model_id, test_store
            )

            # Generate predictions for the testing data then training data
            for store in (test_store, train_store):
                if self.predictor.replace or self.model_evaluator.needs_evaluations(store, model_id):
                    logging.info(
                        "The evaluations needed for matrix %s-%s and model %s"
                        "are not all present in db, so predicting and evaluating",
                        store.uuid,
                        store.matrix_type,
                        model_id
                    )
                    predictions_proba = self.predictor.predict(
                        model_id,
                        store,
                        misc_db_parameters=dict(),
                        train_matrix_columns=train_store.columns(),
                    )

                    self.model_evaluator.evaluate(
                        predictions_proba=predictions_proba,
                        matrix_store=store,
                        model_id=model_id,
                    )
                else:
                    logging.info(
                        "The evaluations needed for matrix %s-%s and model %s are all present"
                        "in db from a previous run (or none needed at all), so skipping!",
                        store.uuid,
                        store.matrix_type,
                        model_id
                    )


__all__ = (
    "IndividualImportanceCalculator",
    "ModelEvaluator",
    "ModelGrouper"
    "ModelTrainer",
    "Predictor",
    "ModelTrainTester"
)
