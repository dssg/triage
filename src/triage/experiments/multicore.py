import logging
import traceback
from functools import partial
from multiprocessing import Pool

from triage.component.catwalk.utils import Batch
from triage.component.catwalk.storage import InMemoryModelStorageEngine

from triage.experiments import ExperimentBase


class MultiCoreExperiment(ExperimentBase):
    def __init__(self, n_processes=1, n_db_processes=1, *args, **kwargs):
        super(MultiCoreExperiment, self).__init__(*args, **kwargs)
        self.n_processes = n_processes
        self.n_db_processes = n_db_processes
        if kwargs['model_storage_class'] == InMemoryModelStorageEngine:
            raise ValueError('''
                InMemoryModelStorageEngine not compatible with MultiCoreExperiment
            ''')

    def generated_chunked_parallelized_results(
        self,
        partially_bound_function,
        tasks,
        n_processes,
        chunksize=1,
    ):
        with Pool(n_processes, maxtasksperchild=1) as pool:
            for result in pool.map(
                partially_bound_function,
                [list(task_batch) for task_batch in Batch(tasks, chunksize)]
            ):
                yield result

    def process_train_tasks(self, train_tasks):
        partial_train_models = partial(
            run_task_with_splatted_arguments,
            self.trainer.process_train_task
        )
        logging.info(
            'Starting parallel training: %s tasks, %s processes',
            len(train_tasks),
            self.n_processes
        )
        model_ids = []
        for model_id in parallelize(
            partial_train_models,
            train_tasks,
            self.n_processes
        ):
            model_ids.append(model_id)
        return model_ids

    def process_model_test_tasks(self, test_tasks):
        partial_test = partial(
            run_task_with_splatted_arguments,
            self.tester.process_model_test_task
        )

        logging.info(
            'Starting parallel testing with %s processes',
            self.n_db_processes
        )
        parallelize(
            partial_test,
            test_tasks,
            self.n_db_processes
        )
        logging.info('Cleaned up concurrent pool')

    def process_query_tasks(self, query_tasks):
        logging.info(
            'Processing query tasks with %s processes',
            self.n_db_processes
        )
        for table_name, tasks in query_tasks.items():
            logging.info('Processing features for %s', table_name)
            self.feature_generator.run_commands(tasks.get('prepare', []))
            partial_insert = partial(insert_into_table, feature_generator=self.feature_generator)

            insert_batches = [list(task_batch) for task_batch in Batch(tasks.get('inserts', []), 25)]
            parallelize(
                partial_insert,
                insert_batches,
                n_processes=self.n_db_processes,
            )
            self.feature_generator.run_commands(tasks.get('finalize', []))
            logging.info('%s completed', table_name)

    def process_matrix_build_tasks(self, matrix_build_tasks):
        partial_build_matrix = partial(
            run_task_with_splatted_arguments,
            self.matrix_builder.build_matrix
        )
        logging.info(
            'Starting parallel matrix building: %s matrices, %s processes',
            len(self.matrix_build_tasks.keys()),
            self.n_processes
        )
        parallelize(
            partial_build_matrix,
            self.matrix_build_tasks.values(),
            self.n_processes
        )


def insert_into_table(insert_statements, feature_generator):
    try:
        logging.info('Beginning insert batch')
        feature_generator.run_commands(insert_statements)
        return True
    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return False


def parallelize(partially_bound_function, tasks, n_processes):
    num_successes = 0
    num_failures = 0
    results = []
    with Pool(n_processes, maxtasksperchild=1) as pool:
        for result in pool.map(partially_bound_function, tasks):
            if result:
                num_successes += 1
            else:
                num_failures += 1
            results.append(result)

        logging.info(
            'Done. successes: %s, failures: %s',
            num_successes,
            num_failures
        )
        return results


def run_task_with_splatted_arguments(task_runner, task):
    try:
        return task_runner(**task)
    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return None
