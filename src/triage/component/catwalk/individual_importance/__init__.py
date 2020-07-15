import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from triage.component.catwalk.utils import save_db_objects
from triage.component.results_schema import IndividualImportance

from .uniform import uniform_distribution


CALCULATE_STRATEGIES = {"uniform": uniform_distribution}


class IndividualImportanceCalculatorNoOp:
    def calculate_and_save_all_methods_and_dates(self, model_id, test_matrix_store):
        logger.notice(
            "No individual feature importance configuration is available, so no individual feature importance will be created"
        )

    def calculate_and_save(self, model_id, test_matrix_store, method, as_of_date):
        logger.notice(
            "No individual feature importance configuration is available, so no individual feature importance will be created"
        )


    def save(self, importance_records, model_id, as_of_date, method_name):
        logger.notice(
            "No individual feature importance configuration is available, so no individual feature importance will be created"
        )



class IndividualImportanceCalculator:
    """Calculates and saves individual importance scores and rankings using different methods

    Args:
        db_engine (sqlalchemy.engine)
        n_ranks (int) Number of ranks to calculate and save. Defaults to 5
        methods (list) Strings corresponding to individual importance methods  ,
            present in CALCULATE_STRATEGIES that should be called.
            Defaults to ['uniform']
        replace (bool) Whether to replace old records or reuse them.
    """

    def __init__(self, db_engine, n_ranks=5, methods=["uniform"], replace=True):
        self.db_engine = db_engine
        self.n_ranks = n_ranks
        self.methods = methods
        self.replace = replace

    def _num_existing_importances(self, model_id, as_of_date, method):
        return [
            row[0]
            for row in self.db_engine.execute(
                """select count(*) from test_results.individual_importances
            where model_id = %s
            and as_of_date = %s
            and method = %s""",
                model_id,
                as_of_date,
                method,
            )
        ][0]

    def _needs_new_importances(self, model_id, as_of_date, method, matrix_store):
        """Determines whether or not importances matching the arguments are present in the database

        To do this, we check how many importances exist for the given arguments
        and compare with the number of importances that the given matrix will produce.

        Args:
            model_id (int) A model id, expected to be present in test_results.models
            as_of_date (datetime or string) The date to produce individual importances as of
            method (string) The name of a method to use to produce individual importances
            matrix_store (catwalk.storage.MatrixStore) The test matrix

        Returns: (bool) whether or not there are fewer importances in the db than the matrix
        """
        existing_importances = self._num_existing_importances(
            model_id, as_of_date, method
        )
        expected_importances = matrix_store.num_entities * self.n_ranks
        logger.debug(
            "model_id=%s/as_of_date=%s/method=%s: found %s importances",
            model_id,
            as_of_date,
            method,
            existing_importances,
        )
        logger.debug(
            "matrix_uuid=%s/n_ranks=%s: expect %s importances",
            matrix_store.uuid,
            self.n_ranks,
            expected_importances,
        )
        return existing_importances < expected_importances

    def calculate_and_save_all_methods_and_dates(self, model_id, test_matrix_store):
        """Calculate and save individual importances for the given model and test matrix

        Args:
            model_id (int) A model id, expected to be present in test_results.models
            test_matrix_store (catwalk.storage.MatrixStore) The test matrix
        """
        for method in self.methods:
            for as_of_date in test_matrix_store.as_of_dates:
                self.calculate_and_save(model_id, test_matrix_store, method, as_of_date)

    def calculate_and_save(self, model_id, test_matrix_store, method, as_of_date):
        """Calculate and save importances for a given model, test matrix, method, and date

        Args:
            model_id (int) A model id, expected to be present in test_results.models
            test_matrix_store (catwalk.storage.MatrixStore) The test matrix
            method_name (string) The name of a method to use to produce individual importances
                Expected to be present in CALCULATE_STRATEGIES
            as_of_date (datetime or string) The date to produce individual importances as of
        """
        if not self.replace and not self._needs_new_importances(
            model_id, as_of_date, method, test_matrix_store
        ):
            logger.debug(
                "Found as many or more individual importances "
                + "for model_id=%s/as_of_date=%s/method=%s, skipping",
                model_id,
                as_of_date,
                method,
            )
        else:
            importance_records = CALCULATE_STRATEGIES[method](
                self.db_engine, model_id, as_of_date, test_matrix_store, self.n_ranks
            )
            self.save(
                importance_records=importance_records,
                model_id=model_id,
                as_of_date=as_of_date,
                method_name=method,
            )

    def save(self, importance_records, model_id, as_of_date, method_name):
        """Saves computed individual feature importance records to the database.
        Will delete any records beforehand matching the model_id, as_of_date, and method_name

        Args:
            importance_records (list) Individual importances.
                Each entry should be a dict with keys 'entity_id' 'feature_name'
                and 'feature_value' corresponding to the correct values for the
                model_id, as_of_date, and method
            model_id (int) A model id, expected to be present in test_results.models
            as_of_date (datetime or string) The as_of_date matching the importance records
            method_name (string) The name of the method that produced the importance records

        """
        self.db_engine.execute(
            """delete from test_results.individual_importances
            where model_id = %s
            and as_of_date = %s
            and method = %s""",
            model_id,
            as_of_date,
            method_name,
        )
        record_stream = (
            IndividualImportance(
                model_id=int(model_id),
                entity_id=int(importance_record["entity_id"]),
                as_of_date=as_of_date,
                feature=importance_record["feature_name"],
                feature_value=importance_record["feature_value"],
                method=method_name,
                importance_score=float(importance_record["score"]),
            )
            for importance_record in importance_records
        )
        save_db_objects(self.db_engine, record_stream)
