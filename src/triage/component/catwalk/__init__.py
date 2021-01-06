"""Main application"""
from .model_trainers import ModelTrainer
from .predictors import Predictor
from .evaluation import ModelEvaluator
from .individual_importance import IndividualImportanceCalculator, IndividualImportanceCalculatorNoOp
from .model_grouping import ModelGrouper
from .subsetters import Subsetter, SubsetterNoOp
from .protected_groups_generators import ProtectedGroupsGenerator, ProtectedGroupsGeneratorNoOp
from .utils import filename_friendly_hash

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from collections import namedtuple

import numpy as np

TaskBatch = namedtuple('TaskBatch', ['parallelizable', 'tasks', 'description'])


class ModelTrainTester:
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
        self.protected_groups_generator = protected_groups_generator
        self.cohort_hash = cohort_hash

    def generate_task_batches(self, splits, grid_config, model_comment=None):
        train_test_tasks = []
        logger.debug(f"Generating train/test tasks for {len(splits)} splits")
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
        logger.verbose("Split train/test tasks into three task batches. - each batch has models from all splits")
        for batch_num, batch in enumerate(batches, 1):
            logger.verbose(f"Batch {batch_num}: {batch.description} ({len(batch.tasks)} tasks total)")



        return batches


    def process_all_batches(self, task_batches):
        for n_batch, batch in enumerate(task_batches, start=1):
            logger.verbose(f"Processing '{batch.description}' [{n_batch} of {len(task_batches)} batches]")
            for n_task, task in enumerate(batch.tasks, start=1):
                logger.verbose(f"Processing task [{n_task} of {len(batch.tasks)}] from {batch.description}")
                self.process_task(**task)
                logger.verbose(f"Task {n_task} from {batch.description} completed")
            logger.success(f"Batch '{batch.description}' completed")

    def process_task(self, test_store, train_store, train_kwargs):
        logger.verbose(f"Training {train_kwargs.get('class_path')}({train_kwargs.get('parameters')}) [{train_kwargs.get('model_hash')}] on train matrix {train_store.uuid}")

        # If the matrices and train labels are OK, train and test the model!
        with self.model_trainer.cache_models(), test_store.cache(), train_store.cache():
            # will cache any trained models until it goes out of scope (at the end of the task)
            # this way we avoid loading the model pickle again for predictions

            # If the train or test design matrix empty, or if the train store only
            # has one label value, skip training the model.
            if train_store.empty:
                logger.notice(
                    f"""Train matrix for split {train_store.uuid} was empty,
                    no point in training this model. Skipping
                    """
                )
                return

            if len(train_store.labels.unique()) == 1:
                logger.notice(
                    f"""Train Matrix for split {train_store.uuid} had only one
                    unique value, no point in training this model. Skipping
                    """
                )
                return

            if test_store.empty:
                logger.notice(
                    f"""Test matrix for uuid {test_store.uuid}
                    was empty, no point in generating predictions. Not processing train/test task.
                    """
                )
                return

            model_id = self.model_trainer.process_train_task(**train_kwargs)

            if not model_id:
                logger.warning("Training unsuccessful for {train_kwargs.get('class_path')}({train_kwargs.get('parameters')}) [{train_kwargs.get('model_hash')}] on train matrix {train_store.uuid}. "
                               "No model id returned.  Not attempting to test it")
                return

            logger.success(f"Trained model id {model_id}: {train_kwargs.get('class_path')}({train_kwargs.get('parameters')}) [{train_kwargs.get('model_hash')}] on train matrix {train_store.uuid}. ")

            # Storing individual importances (if any)
            self.individual_importance_calculator.calculate_and_save_all_methods_and_dates(
                model_id, test_store
            )

            as_of_dates = test_store.as_of_dates
            logger.debug(
                f"Testing and evaluating model {model_id}  {train_kwargs.get('class_path')}({train_kwargs.get('parameters')}) [{train_kwargs.get('model_hash')}] "
                f"on test matrix {test_store.uuid}. ")
            logger.spam(f"as_of_times min: {min(as_of_dates)} max: {max(as_of_dates)} num: {len(as_of_dates)}")


            # Generate predictions for the testing data then training data
            for store in (test_store, train_store):
                if self.replace or self.model_evaluator.needs_evaluations(store, model_id):
                    logger.spam(
                        f"Generating new predictions for "
                        f"{store.matrix_type.string_name} matrix {store.uuid}, and model {model_id} to make evaluation",
                    )

                    predictions_proba = self.predictor.predict(
                        model_id,
                        store,
                        misc_db_parameters=dict(),
                        train_matrix_columns=train_store.columns(),
                    )

                    logger.debug(f"Predictions generated for {store.matrix_type.string_name} matrix {store.uuid} using model {model_id}")


                    protected_df = self.protected_groups_generator.as_dataframe(
                        as_of_dates=store.as_of_dates,
                        cohort_hash=self.cohort_hash,
                    )

                    logger.spam(
                        f"Evaluating model {model_id} on {store.matrix_type.string_name} matrix {store.uuid} "
                    )

                    self.model_evaluator.evaluate(
                        predictions_proba=predictions_proba,
                        matrix_store=store,
                        model_id=model_id,
                        subset=None,
                        protected_df=protected_df
                    )

                    logger.info(
                        f"Model {model_id} evaluation on {store.matrix_type.string_name} matrix {store.uuid} completed."
                    )

                else:
                    logger.notice(
                        f"The evaluations needed for {store.matrix_type.string_name} matrix {store.uuid} and "
                        f"model {model_id} are all present"
                        f"in db from a previous run (or none needed at all), so skipping!",
                    )


                for subset in self.subsets:
                    subset_hash = filename_friendly_hash(subset)
                    if self.replace or self.model_evaluator.needs_evaluations(store, model_id, subset_hash):

                        logger.spam(
                            f"Evaluating {store.matrix_type.string_name} matrix {store.uuid}, subset {subset_hash}, and model {model_id}"
                        )


                        self.model_evaluator.evaluate(
                            predictions_proba=predictions_proba,
                            matrix_store=store,
                            model_id=model_id,
                            subset=subset,
                            protected_df=protected_df
                        )

                        logger.info(
                            f"Model {model_id} evaluation on subset {filename_friendly_hash(subset)} of {store.matrix_type.string_name} matrix {store.uuid} completed."
                        )

                    else:
                        logger.notice(
                            f"The evaluations needed for {store.matrix_type.string_name} matrix {store.uuid}, subset {filename_friendly_hash(subset)}, and "
                            f"model {model_id} are all present"
                            f"in db from a previous run (or none needed at all), so skipping!",
                        )


__all__ = (
    "IndividualImportanceCalculator",
    "ModelEvaluator",
    "ModelGrouper"
    "ModelTrainer",
    "Predictor",
    "ModelTrainTester",
    "Subsetter",
)
