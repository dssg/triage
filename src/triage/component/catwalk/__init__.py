"""Main application"""
from .model_trainers import ModelTrainer
from .predictors import Predictor
from .evaluation import ModelEvaluator
from .individual_importance import IndividualImportanceCalculator
from .model_grouping import ModelGrouper
from .subsetters import Subsetter
from .protected_groups_generators import ProtectedGroupsGenerator, ProtectedGroupsGeneratorNoOp
from .utils import filename_friendly_hash
import logging
from collections import namedtuple

import numpy
import pandas

TaskBatch = namedtuple('TaskBatch', ['parallelizable', 'tasks', 'description'])


class ModelTrainTester(object):
    def __init__(
        self,
        matrix_storage_engine,
        model_trainer,
        model_evaluator,
        individual_importance_calculator,
        predictor,
        subsets,
        protected_groups_generator,
        cohort_hash=None,
        replace=True
    ):
        self.matrix_storage_engine = matrix_storage_engine
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.individual_importance_calculator = individual_importance_calculator
        self.predictor = predictor
        self.subsets = subsets
        self.replace = replace
        self.protected_groups_generator = protected_groups_generator or ProtectedGroupsGeneratorNoOp()
        self.cohort_hash = cohort_hash

    def generate_task_batches(self, splits, grid_config, model_comment=None):
        train_test_tasks = []
        logging.info("Generating train/test tasks for %s splits", len(splits))
        for split in splits:
            train_store = self.matrix_storage_engine.get_store(split["train_uuid"])
            train_tasks = self.model_trainer.generate_train_tasks(
                grid_config=grid_config,
                misc_db_parameters=dict(test=False, model_comment=model_comment),
                matrix_store=train_store
            )

            for test_matrix_def, test_uuid in zip(
                split["test_matrices"], split["test_uuids"]
            ):
                test_store = self.matrix_storage_engine.get_store(test_uuid)

                for train_task in train_tasks:
                    train_test_tasks.append(
                        {
                            "test_store": test_store,
                            "train_store": train_store,
                            "train_kwargs": train_task,
                        }
                    )
        return self.order_and_batch_tasks(train_test_tasks)


    def order_and_batch_tasks(self, tasks):
        batches = (
            TaskBatch(
                parallelizable=True,
                tasks=[],
                description="Baselines or simple classifiers (e.g. DecisionTree, SLR)"
            ),
            TaskBatch(
                parallelizable=False,
                tasks=[],
                description="Heavyweight classifiers with n_jobs set to -1."
            ),
            TaskBatch(
                parallelizable=True,
                tasks=[],
                description="All classifiers not found in one of the other batches (e.g. gradient boosting)."
            ),
        )
         
        for task in tasks:
            if task['train_kwargs']['class_path'].startswith('triage.component.catwalk.baselines') \
                    or task['train_kwargs']['class_path'] in (
                    'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression',
                    'sklearn.tree.DecisionTreeClassifier',
                    'sklearn.dummy.DummyClassifier'
                    ):
                # First priority: baselines or simple, effective classifiers
                batches[0].tasks.append(task)
            elif task['train_kwargs']['parameters'].get('n_jobs', None) == -1:
                # Second priority: heavyweight classifiers that we use the whole machines for
                batches[1].tasks.append(task)
            else:
                # Last priority: Everything else. Maybe these are slow/non-parallelizable
                batches[2].tasks.append(task)
        logging.info("Split train/test tasks into three task batches. - each batch has models from all splits")
        for batch_num, batch in enumerate(batches, 1):
            logging.info("Batch %s: %s (%s tasks total)", batch_num, batch.description, len(batch.tasks))
        return batches


    def process_all_batches(self, task_batches):
        # In the simple loop version here we ignore parallelizability and do everything serially
        for batch in task_batches:
            for task in batch.tasks:
                self.process_task(**task)

    def process_task(self, test_store, train_store, train_kwargs):
        logging.info("Beginning train task %s", train_kwargs)

        # If the matrices and train labels are OK, train and test the model!
        with self.model_trainer.cache_models(), test_store.cache(), train_store.cache():
            # will cache any trained models until it goes out of scope (at the end of the task)
            # this way we avoid loading the model pickle again for predictions

            # If the train or test design matrix empty, or if the train store only
            # has one label value, skip training the model.
            if train_store.empty:
                logging.warning(
                    """Train matrix for split %s was empty,
                no point in training this model. Skipping
                """,
                    train_store.uuid
                )
                return
            if len(train_store.labels.unique()) == 1:
                logging.warning(
                    """Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                """,
                    train_store.uuid,
                )
                return
            if test_store.empty:
                logging.warning(
                    """Test matrix for uuid %s
                was empty, no point in generating predictions. Not processing train/test task.
                """,
                    test_store.uuid,
                )
                return

            model_id = self.model_trainer.process_train_task(**train_kwargs)
            if not model_id:
                logging.warning("No model id returned from ModelTrainer.process_train_task, "
                                "training unsuccessful. Not attempting to test")
                return
            logging.info("Trained task %s and got model id %s", train_kwargs, model_id)
            as_of_dates = test_store.as_of_dates
            logging.info(
                "Testing and scoring model id %s with test matrix %s. "
                "as_of_times min: %s max: %s num: %s",
                model_id,
                test_store.uuid,
                min(as_of_dates),
                max(as_of_dates),
                len(as_of_dates),
            )

            self.individual_importance_calculator.calculate_and_save_all_methods_and_dates(
                model_id, test_store
            )

            # Generate predictions for the testing data then training data
            for store in (test_store, train_store):
                predictions_proba = numpy.array(None)
                protected_df = None
                if self.replace:
                    logging.info(
                        "Replace flag set; generating new predictions and evaluations for"
                        "matrix %s-%s, and model %s",
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

                for subset in self.subsets:
                    if self.replace or self.model_evaluator.needs_evaluations(
                        store, model_id, filename_friendly_hash(subset)
                    ):
                        logging.info(
                            "Evaluating matrix %s-%s, subset %s, and model %s",
                            store.uuid,
                            store.matrix_type,
                            filename_friendly_hash(subset),
                            model_id,
                        )

                        if not predictions_proba.any():
                            logging.info(
                                "Generating new predictions for"
                                "matrix %s-%s, and model %s to make evaluation",
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
                        if protected_df is None:
                            protected_df = self.protected_groups_generator.as_dataframe(
                                as_of_dates=store.as_of_dates,
                                cohort_hash=self.cohort_hash,
                            )

                        self.model_evaluator.evaluate(
                            predictions_proba=predictions_proba,
                            matrix_store=store,
                            model_id=model_id,
                            subset=subset,
                            protected_df=protected_df
                        )

                    else:
                        logging.info(
                            "The evaluations needed for matrix %s-%s, subset %s, and "
                            "model %s are all present"
                            "in db from a previous run (or none needed at all), so skipping!",
                            store.uuid,
                            store.matrix_type,
                            filename_friendly_hash(subset),
                            model_id
                        )
                self.predictor.update_db_with_ranks(model_id, store.uuid, store.matrix_type)


__all__ = (
    "IndividualImportanceCalculator",
    "ModelEvaluator",
    "ModelGrouper"
    "ModelTrainer",
    "Predictor",
    "ModelTrainTester",
    "Subsetter",
)
