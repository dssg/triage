import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import traceback
from functools import partial
from pebble import ProcessPool
from multiprocessing.reduction import ForkingPickler

from triage.component.catwalk.utils import Batch
from triage.component.catwalk import BatchKey

from triage.experiments import ExperimentBase


class MultiCoreExperiment(ExperimentBase):
    def __init__(self, config, db_engine, *args, n_processes=1, n_bigtrain_processes=1, n_db_processes=1, **kwargs):
        """
        Args:
            config (dict)
            db_engine (sqlalchemy engine)
            n_processes (int) How many parallel processes to use for most CPU-bound tasks.
                Logistic regression and decision trees fall under this category.
                Usually good to set to the # of cores on the machine.
            n_bigtrain_processes (int) How many parallel processes to use for memory-intensive tasks
                Random forests and extra trees fall under this category.
                Usually good to start at 1, but can be increased if you have available memory.
            n_db_processes (int) How many parallel processes to use for database IO-intensive tasks.
                Cohort creation, label creation, and feature creation fall under this category. 
        """
        try:
            ForkingPickler.dumps(db_engine)
        except Exception as exc:
            raise TypeError(
                "multiprocessing is unable to pickle passed SQLAlchemy engine. "
                "use triage.create_engine instead when running MultiCoreExperiment: "
                "(e.g. from triage import create_engine)"
            ) from exc

        super(MultiCoreExperiment, self).__init__(config, db_engine, *args, **kwargs)
        if n_processes < 1:
            raise ValueError("n_processes must be 1 or greater")
        if n_db_processes < 1:
            raise ValueError("n_db_processes must be 1 or greater")
        if n_bigtrain_processes < 1:
            raise ValueError("n_bigtrain_processes must be 1 or greater")
        if n_db_processes == 1 and n_processes == 1 and n_bigtrain_processes == 1:
            logger.notice(
                "Both n_processes and n_db_processes were set to 1. "
                "If you only wish to use one process to run the experiment, "
                "consider using the SingleThreadedExperiment class instead"
            )
        self.n_processes = n_processes
        self.n_db_processes = n_db_processes
        self.n_bigtrain_processes = n_bigtrain_processes
        self.n_processes_lookup = {
            BatchKey.QUICKTRAIN: self.n_processes,
            BatchKey.BIGTRAIN: self.n_bigtrain_processes,
            BatchKey.MAYBETRAIN: self.n_processes
        }


    def generated_chunked_parallelized_results(
        self, partially_bound_function, tasks, n_processes, chunksize=1
    ):
        with ProcessPool(n_processes, max_tasks=1) as pool:
            future = pool.map(
                partially_bound_function,
                [list(task_batch) for task_batch in Batch(tasks, chunksize)],
            )
            iterator = future.result()
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    break
                except Exception:
                    logger.exception('Child failure')

    def process_train_test_batches(self, batches):
        partial_test = partial(
            run_task_with_splatted_arguments, self.model_train_tester.process_task
        )

        for batch in batches:
            logger.info(
                f"Starting parallelizable batch train/testing with {len(batch.tasks)} tasks, {self.n_processes_lookup[batch.key]} processes",
            )
            parallelize(partial_test, batch.tasks, self.n_processes_lookup[batch.key])

    def process_query_tasks(self, query_tasks):
        logger.info("Processing query tasks with %s processes", self.n_db_processes)
        for table_name, tasks in query_tasks.items():
            logger.info("Processing features for %s", table_name)
            self.feature_generator.run_commands(tasks.get("prepare", []))
            partial_insert = partial(
                insert_into_table, feature_generator=self.feature_generator
            )

            insert_batches = [
                list(task_batch) for task_batch in Batch(tasks.get("inserts", []), 25)
            ]
            parallelize(partial_insert, insert_batches, n_processes=self.n_db_processes)
            self.feature_generator.run_commands(tasks.get("finalize", []))
            logger.info(f"{table_name} completed")

    def process_matrix_build_tasks(self, matrix_build_tasks):
        partial_build_matrix = partial(
            run_task_with_splatted_arguments, self.matrix_builder.build_matrix
        )
        logger.info(
            f"Starting parallel matrix building: {len(self.matrix_build_tasks.keys())} matrices, {self.n_processes} processes",
        )
        parallelize(
            partial_build_matrix, self.matrix_build_tasks.values(), self.n_processes
        )

    def process_subset_tasks(self, subset_tasks):
        partial_subset = partial(
            run_task_with_splatted_arguments, self.subsetter.process_task
        )

        logger.info(
            f"Starting parallel subset creation: {len(subset_tasks)} subsets, {self.n_db_processes} processes",
        )
        parallelize(
            partial_subset, subset_tasks, self.n_db_processes
        )


def insert_into_table(insert_statements, feature_generator):
    try:
        logger.info("Beginning insert batch")
        feature_generator.run_commands(insert_statements)
        return True
    except Exception:
        logger.exception("Child error")
        return False


def parallelize(partially_bound_function, tasks, n_processes):
    num_successes = 0
    num_failures = 0
    results = []
    with ProcessPool(n_processes, max_tasks=1) as pool:
        future = pool.map(partially_bound_function, tasks)
        iterator = future.result()
        results = []
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except Exception:
                logger.exception('Child failure')
                num_failures += 1
            else:
                results.append(result)
                num_successes += 1

        logger.info("Done. successes: %s, failures: %s", num_successes, num_failures)
        return results


def run_task_with_splatted_arguments(task_runner, task):
    try:
        return task_runner(**task)
    except Exception:
        logger.exception("Child error")
        return None
