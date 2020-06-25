import logging
import math

import numpy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_

from .utils import db_retry, retrieve_model_hash_from_id, save_db_objects, sort_predictions_and_labels, AVAILABLE_TIEBREAKERS
from triage.component.results_schema import Model
from triage.util.db import scoped_session
from triage.util.random import generate_python_random_seed
import ohio.ext.pandas
import pandas


class ModelNotFoundError(ValueError):
    pass


class Predictor(object):
    expected_matrix_ts_format = "%Y-%m-%d %H:%M:%S"
    available_tiebreakers = AVAILABLE_TIEBREAKERS

    def __init__(
        self,
        model_storage_engine,
        db_engine,
        rank_order,
        replace=True,
        save_predictions=True
    ):
        """Encapsulates the task of generating predictions on an arbitrary
        dataset and storing the results

        Args:
            model_storage_engine (catwalk.storage.ModelStorageEngine)
            db_engine (sqlalchemy.engine)
            rank_order

        """
        self.model_storage_engine = model_storage_engine
        self.db_engine = db_engine
        self.rank_order = rank_order
        self.replace = replace
        self.save_predictions = save_predictions

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    @db_retry
    def load_model(self, model_id):
        """Downloads the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database

        Returns:
            A python object which implements .predict()
        """

        model_hash = retrieve_model_hash_from_id(self.db_engine, model_id)
        logging.info("Checking for model_hash %s in store", model_hash)
        if self.model_storage_engine.exists(model_hash):
            return self.model_storage_engine.load(model_hash)

    @db_retry
    def delete_model(self, model_id):
        """Deletes the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database
        """
        model_hash = retrieve_model_hash_from_id(self.db_engine, model_id)
        self.model_storage_engine.delete(model_hash)

    @db_retry
    def _existing_predictions(self, Prediction_obj, session, model_id, matrix_store):
        return (
            session.query(Prediction_obj)
            .filter_by(model_id=model_id)
            .filter(Prediction_obj.as_of_date.in_(matrix_store.as_of_dates))
        )

    @db_retry
    def needs_predictions(self, matrix_store, model_id):
        """Returns whether or not the given matrix and model are lacking any predictions

        Args:
            matrix_store (triage.component.catwalk.storage.MatrixStore) A matrix with metadata
            model_id (int) A database ID of a model

        The way we check is by grabbing all the distinct as-of-dates in the predictions table
        for this model and matrix. If there are more as-of-dates defined in the matrix's metadata
        than are in the table, we need predictions
        """
        if not self.save_predictions:
            return False
        session = self.sessionmaker()
        prediction_obj = matrix_store.matrix_type.prediction_obj
        as_of_dates_in_db = set(
            as_of_date.date()
            for (as_of_date,) in session.query(prediction_obj).filter_by(
                model_id=model_id,
                matrix_uuid=matrix_store.uuid
            ).distinct(prediction_obj.as_of_date).values("as_of_date")
        )
        as_of_dates_needed = set(matrix_store.as_of_dates)
        needed = bool(as_of_dates_needed - as_of_dates_in_db)
        session.close()
        return needed

    @db_retry
    def _load_saved_predictions(self, existing_predictions, matrix_store):
        index = matrix_store.index
        score_lookup = {}
        for prediction in existing_predictions:
            score_lookup[
                (prediction.entity_id, prediction.as_of_date.date())
            ] = prediction.score
        score_iterator = (
            score_lookup[(entity_id, dt.date())] for (entity_id, dt) in index
        )
        return numpy.fromiter(score_iterator, float)

    @db_retry
    def _write_predictions_to_db(
        self,
        model_id,
        matrix_store,
        predictions,
        labels,
        misc_db_parameters,
        Prediction_obj,
    ):
        """Writes given predictions to database

        entity_ids, predictions, labels are expected to be in the same order

        Args:
            model_id (int) the id of the model associated with the given predictions
            matrix_store (catwalk.storage.MatrixStore) the matrix and metadata
            entity_ids (iterable) entity ids that predictions were made on
            predictions (iterable) predicted values
            labels (iterable) labels of prediction set (int) the id of the model
            to predict based off of
            Prediction_obj (TrainPrediction or TestPrediction) table to store predictions to

        """
        try:
            session = self.sessionmaker()
            self._existing_predictions(
                Prediction_obj, session, model_id, matrix_store
            ).delete(synchronize_session=False)
            session.expire_all()
            session.commit()
        finally:
            session.close()
        test_label_timespan = matrix_store.metadata["label_timespan"]

        record_stream = (
            Prediction_obj(
                model_id=int(model_id),
                entity_id=int(entity_id),
                as_of_date=as_of_date,
                score=float(score),
                label_value=int(label) if not math.isnan(label) else None,
                matrix_uuid=matrix_store.uuid,
                test_label_timespan=test_label_timespan,
                **misc_db_parameters
            )
            for ((entity_id, as_of_date), score, label) in zip(
                matrix_store.index, predictions, labels
            )
        )
        save_db_objects(self.db_engine, record_stream)

    def _write_metadata_to_db(self, model_id, matrix_uuid, matrix_type, random_seed):
        orm_obj = matrix_type.prediction_metadata_obj(
            model_id=model_id,
            matrix_uuid=matrix_uuid,
            tiebreaker_ordering=self.rank_order,
            random_seed=random_seed,
            predictions_saved=self.save_predictions,
        )
        session = self.sessionmaker()
        session.merge(orm_obj)
        session.commit()
        session.close()

    def _needs_ranks(self, model_id, matrix_uuid, matrix_type):
        if self.replace:
            logging.debug("Replace flag set, will compute and store ranks regardless")
            return True
        with scoped_session(self.db_engine) as session:
            # if the metadata is different (e.g. they changed the rank order)
            # or there are any null ranks we need to rank
            metadata_matches = session.query(session.query(matrix_type.prediction_metadata_obj).filter_by(
                model_id=model_id,
                matrix_uuid=matrix_uuid,
                tiebreaker_ordering=self.rank_order,
            ).exists()).scalar()
            if not metadata_matches:
                logging.debug("prediction metadata does not match what is in configuration"
                              ", will compute and store ranks")
                return True

            any_nulls_in_ranks = session.query(session.query(matrix_type.prediction_obj)\
                .filter(
                    matrix_type.prediction_obj.model_id == model_id,
                    matrix_type.prediction_obj.matrix_uuid == matrix_uuid,
                    or_(
                        matrix_type.prediction_obj.rank_abs_no_ties == None,
                        matrix_type.prediction_obj.rank_abs_with_ties == None,
                        matrix_type.prediction_obj.rank_pct_no_ties == None,
                        matrix_type.prediction_obj.rank_pct_with_ties == None,
                    )
                ).exists()).scalar()
            if any_nulls_in_ranks:
                logging.debug("At least one null in rankings in predictions table",
                              ", will compute and store ranks")
                return True
        logging.debug("No need to recompute prediction ranks")
        return False

    def update_db_with_ranks(self, model_id, matrix_uuid, matrix_type):
        """Update predictions table with rankings, both absolute and percentile.
                random_seed=postgres_random_seed,
        All entities should have different ranks, so to break ties:
        - abs_rank uses the 'row_number' function, so ties are broken by the database ordering
            session.close()
        - pct_rank uses the output of the abs_rank to compute percentiles
          (as opposed to raw scores), so it inherits the tie-breaking from abs_rank
        Args:
            model_id (int) the id of the model associated with the given predictions
            matrix_uuid (string) the uuid of the prediction matrix
        """
        if not self.save_predictions:
            logging.info("save_predictions is set to False so there are no predictions to rank")
            return
        logging.info(
            'Beginning ranking of new Predictions for model %s, matrix %s',
            model_id,
            matrix_uuid
        )

        # retrieve a dataframe with only the data we need to rank
        ranking_df = pandas.DataFrame.pg_copy_from(
            f"""select entity_id, score, as_of_date, label_value
            from {matrix_type.string_name}_results.predictions
            where model_id = {model_id} and matrix_uuid = '{matrix_uuid}'
            """, connectable=self.db_engine)

        sort_seed = None
        if self.rank_order == 'random':
            with scoped_session(self.db_engine) as session:
                sort_seed = session.query(Model).get(model_id).random_seed
                if not sort_seed:
                    sort_seed = generate_python_random_seed()

        sorted_predictions, sorted_labels, sorted_arrays = sort_predictions_and_labels(
            predictions_proba=ranking_df['score'],
            labels=ranking_df['label_value'],
            tiebreaker=self.rank_order,
            sort_seed=sort_seed,
            parallel_arrays=(ranking_df['entity_id'], ranking_df['as_of_date']),
        )
        ranking_df['score'] = sorted_predictions.values
        ranking_df['as_of_date'] = pandas.to_datetime(sorted_arrays[1].values)
        ranking_df['label_value'] = sorted_labels.values
        ranking_df['entity_id'] = sorted_arrays[0].values
        # at this point, we have the same dataframe that we loaded from postgres,
        # but sorted based on score and the self.rank_order.

        # Now we can generate ranks using pandas and only using the 'score' column because
        # our secondary ordering is baked in, enabling the 'first' method to break ties.
        ranking_df['rank_abs_no_ties'] = ranking_df['score'].rank(ascending=False, method='first')
        ranking_df['rank_abs_with_ties'] = ranking_df['score'].rank(ascending=False, method='min')
        ranking_df['rank_pct_no_ties'] = numpy.array([1 - (rank - 1) / len(ranking_df) for rank in ranking_df['rank_abs_no_ties']])
        ranking_df['rank_pct_with_ties'] = ranking_df['score'].rank(method='min', pct=True)

        # with our rankings computed, update these ranks into the existing rows
        # in the predictions table
        temp_table_name = f"ranks_mod{model_id}_mat{matrix_uuid}"
        ranking_df.pg_copy_to(temp_table_name, self.db_engine)
        self.db_engine.execute(f"""update {matrix_type.string_name}_results.predictions as p
            set rank_abs_no_ties = tt.rank_abs_no_ties,
            rank_abs_with_ties = tt.rank_abs_with_ties,
            rank_pct_no_ties = tt.rank_pct_no_ties,
            rank_pct_with_ties = tt.rank_pct_with_ties
            from {temp_table_name} as tt
            where tt.entity_id = p.entity_id
            and p.matrix_uuid = '{matrix_uuid}'
            and p.model_id = {model_id}
            and p.as_of_date = tt.as_of_date
                               """)
        self.db_engine.execute(f"drop table {temp_table_name}")
        self._write_metadata_to_db(
            model_id=model_id,
            matrix_uuid=matrix_uuid,
            matrix_type=matrix_type,
            random_seed=sort_seed,
        )
        logging.info(
            'Completed ranking of new Predictions for model %s, matrix %s',
            model_id,
            matrix_uuid
        )

    def predict(self, model_id, matrix_store, misc_db_parameters, train_matrix_columns):
        """Generate predictions and store them in the database

        Args:
            model_id (int) the id of the trained model to predict based off of
            matrix_store (catwalk.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata
            misc_db_parameters (dict): attributes and values to add to each
                TrainPrediction or TestPrediction object in the results schema
            train_matrix_columns (list): The order of columns that the model
                was trained on

        Returns:
            (numpy.Array) the generated prediction values
        """
        # Setting the Prediction object type - TrainPrediction or TestPrediction
        matrix_type = matrix_store.matrix_type

        if not self.replace:
            logging.info(
                "Replace flag not set for model id %s, matrix %s, looking for old predictions",
                model_id,
                matrix_store.uuid,
            )
            try:
                session = self.sessionmaker()
                existing_predictions = self._existing_predictions(
                    matrix_type.prediction_obj, session, model_id, matrix_store
                )
                if existing_predictions.count() == len(matrix_store.index):
                    logging.info(
                        "Found predictions for model id %s, matrix %s, returning saved versions",
                        model_id,
                        matrix_store.uuid,
                    )
                    return self._load_saved_predictions(existing_predictions, matrix_store)
            finally:
                session.close()

        model = self.load_model(model_id)
        logging.info("Loaded model %s", model_id)
        if not model:
            raise ModelNotFoundError("Model id {} not found".format(model_id))

        # Labels are popped from matrix (IE, they are removed and returned)
        labels = matrix_store.labels

        predictions_proba = model.predict_proba(
            matrix_store.matrix_with_sorted_columns(train_matrix_columns)
        )
        logging.info(
            "Generated predictions for model %s, matrix %s", model_id, matrix_store.uuid
        )
        if self.save_predictions:
            logging.info(
                "Writing predictions for model %s, matrix %s to database",
                model_id,
                matrix_store.uuid,
            )
            self._write_predictions_to_db(
                model_id,
                matrix_store,
                predictions_proba[:, 1],
                labels,
                misc_db_parameters,
                matrix_type.prediction_obj,
            )
            logging.info(
                "Wrote predictions for model %s, matrix %s to database",
                model_id,
                matrix_store.uuid,
            )
        else:
            logging.info(
                "Skipping prediction database sync for model %s, matrix %s because "
                "save_predictions was marked False",
                model_id,
                matrix_store.uuid,
            )
            self._write_metadata_to_db(
                model_id=model_id,
                matrix_uuid=matrix_store.uuid,
                matrix_type=matrix_type,
                random_seed=None,
            )

        return predictions_proba[:, 1]
