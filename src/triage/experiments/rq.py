import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)
import time
from triage.component.catwalk.utils import Batch
from triage.experiments import ExperimentBase

try:
    from rq import Queue
except ImportError:
    logger.error(
        "rq not available. To use RQExperiment, install triage with the RQ extension: "
        "pip install triage[rq]"
    )
    raise


DEFAULT_TIMEOUT = (
    "365d"
)  # We want to basically invalidate RQ's timeouts by setting them each to one year


class RQExperiment(ExperimentBase):
    """An experiment that uses the python-rq library to enqueue tasks and wait for them to finish.

    http://python-rq.org/

    For this experiment to complete, you need some amount of RQ workers running the Triage codebase
    (either on the same machine as the experiment or elsewhere),
    and a Redis instance that both the experiment process and RQ workers can access.

    Args:
        redis_connection (redis.connection): A connection to a Redis instance that
            some rq workers can also access
        sleep_time (int, default 5) How many seconds the process should sleep while
            waiting for RQ results
        queue_kwargs (dict, default {}) Any extra keyword arguments to pass to Queue creation
    """

    def __init__(
        self, redis_connection, sleep_time=5, queue_kwargs=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.redis_connection = redis_connection
        if queue_kwargs is None:
            queue_kwargs = {}
        self.queue = Queue(connection=self.redis_connection, **queue_kwargs)
        self.sleep_time = sleep_time

    def wait_for(self, jobs):
        """Wait for a list of jobs to complete

        Will run until all jobs are either finished or failed.

        Args:
            jobs (list of rq.Job objects)

        Returns: (list) of job return values
        """
        while True:
            num_done = sum(1 for job in jobs if job.is_finished)
            num_failed = sum(1 for job in jobs if job.is_failed)
            num_pending = sum(
                1 for job in jobs if not job.is_finished and not job.is_failed
            )
            logger.debug(
                f"Report: jobs {num_done} done, {num_failed} failed, {num_pending} pending",
            )
            if num_pending == 0:
                logger.verbose("All jobs completed or failed, returning")
                return [job.result for job in jobs]
            else:
                logger.spam("Sleeping for {self.sleep_time} seconds")
                time.sleep(self.sleep_time)

    def process_query_tasks(self, query_tasks):
        """Run queries by table

        Will run preparation (e.g. create table) and finalize (e.g. create index) tasks
        in the main process,
        but delegate inserts to rq Jobs in batches of 25

        Args: query_tasks (dict) - keys should be table names and values should be dicts.
            Each inner dict should have up to three keys, each with a list of queries:
            'prepare' (setting up the table),
            'inserts' (insert commands to populate the table),
            'finalize' (finishing table setup after all inserts have run)

            Example: {
                'table_one': {
                    'prepare': ['create table table_one (col1 varchar)'],
                    'inserts': [
                        'insert into table_one values (\'a\')',
                        'insert into table_one values (\'b'\')'
                    ]
                    'finalize': ['create index on table_one (col1)']
                }
            }
        """
        for table_name, tasks in query_tasks.items():
            logger.spam(f"Processing features for {table_name}")
            self.feature_generator.run_commands(tasks.get("prepare", []))

            insert_batches = [
                list(task_batch) for task_batch in Batch(tasks.get("inserts", []), 25)
            ]

            jobs = [
                self.queue.enqueue(
                    self.feature_generator.run_commands,
                    insert_batch,
                    job_timeout=DEFAULT_TIMEOUT,
                    result_ttl=DEFAULT_TIMEOUT,
                    ttl=DEFAULT_TIMEOUT,
                )
                for insert_batch in insert_batches
            ]

            self.wait_for(jobs)

            self.feature_generator.run_commands(tasks.get("finalize", []))
            logger.debug(f"{table_name} completed")

    def process_matrix_build_tasks(self, matrix_build_tasks):
        """Run matrix build tasks using RQ

        Args:
            matrix_build_tasks (dict) Keys should be matrix uuids (though not used here),
                values should be dictionaries suitable as kwargs for sending
                to self.matrix_builder.build_matrix

        Returns: (list) of job results for each given task
        """
        jobs = [
            self.queue.enqueue(
                self.matrix_builder.build_matrix,
                job_timeout=DEFAULT_TIMEOUT,
                result_ttl=DEFAULT_TIMEOUT,
                ttl=DEFAULT_TIMEOUT,
                **build_task
            )
            for build_task in matrix_build_tasks.values()
        ]
        return self.wait_for(jobs)

    def process_train_test_batches(self, train_test_batches):
        """Run train tasks using RQ

        Args:
            train_tasks (list) of dictionaries, each representing kwargs suitable
                for self.trainer.process_train_task
        Returns: (list) of job results for each given task
        """
        jobs = [
            self.queue.enqueue(
                self.model_train_tester.process_task,
                job_timeout=DEFAULT_TIMEOUT,
                result_ttl=DEFAULT_TIMEOUT,
                ttl=DEFAULT_TIMEOUT,
                **task
            )
            for batch in train_test_batches
            for task in batch.tasks
        ]
        return self.wait_for(jobs)

    def process_subset_tasks(self, subset_tasks):
        """Run subset tasks using RQ

        Args:
            subset_tasks (list) of dictionaries, each representing kwargs suitable
                for self.subsetter.process_task
        Returns: (list) of job results for each given task
        """
        jobs = [
            self.queue.enqueue(
                self.subsetter.process_task,
                job_timeout=DEFAULT_TIMEOUT,
                result_ttl=DEFAULT_TIMEOUT,
                ttl=DEFAULT_TIMEOUT,
                **task
            )
            for task in subset_tasks
        ]
        return self.wait_for(jobs)
