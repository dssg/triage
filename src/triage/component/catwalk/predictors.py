import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import math

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_
from sklearn.utils import parallel_backend

from .utils import db_retry, retrieve_model_hash_from_id, save_db_objects, sort_predictions_and_labels, AVAILABLE_TIEBREAKERS
from triage.component.results_schema import Model
from triage.util.db import scoped_session
from triage.util.random import generate_python_random_seed
import ohio.ext.pandas
import pandas as pd


class Predictor:
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
        logger.spam(f"Checking for model_hash {model_hash} in store")
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
        logger.debug(f"Looking for existing predictions for model {model_id} on {matrix_store.matrix_type.string_name} matrix [{matrix_store.uuid}]")
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
        return np.fromiter(score_iterator, float)

    @db_retry
    def _write_predictions_to_db(
        self,
        model_id,
        matrix_store,
        df,
        misc_db_parameters,
        Prediction_obj,
    ):
        """Writes given predictions to database

        entity_ids, predictions, labels are expected to be in the same order

        Args:
            model_id (int) the id of the model associated with the given predictions
            matrix_store (catwalk.storage.MatrixStore) the matrix and metadata
            df (pd.DataFrame) with the following columns entity_id, as_of_date, score, label_value and rank_abs_no_ties, rank_abs_with_ties, rank_pct_no_ties, rank_pct_with_ties
    predictions (iterable) predicted values
            Prediction_obj (TrainPrediction or TestPrediction) table to store predictions to

        """
        try:
            session = self.sessionmaker()
            existing_predictions = self._existing_predictions(
                Prediction_obj, session, model_id, matrix_store
            )
            if existing_predictions.count() > 0:
                existing_predictions.delete(synchronize_session=False)
                logger.info(f"Found old predictions for model {model_id} on {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid}. Those predictions were deleted.")
            session.expire_all()
            session.commit()
        finally:
            session.close()
        test_label_timespan = matrix_store.metadata["label_timespan"]

        record_stream = (
            Prediction_obj(
                model_id=int(model_id),
                entity_id=int(row.entity_id),
                as_of_date=row.as_of_date,
                score=float(row.score),
                label_value=int(row.label_value) if not math.isnan(row.label_value) else None,
                rank_abs_no_ties = int(row.rank_abs_no_ties),
                rank_abs_with_ties = int(row.rank_abs_with_ties),
                rank_pct_no_ties = row.rank_pct_no_ties,
                rank_pct_with_ties = row.rank_pct_with_ties,
                matrix_uuid=matrix_store.uuid,
                test_label_timespan=test_label_timespan,
                **misc_db_parameters
            ) for row in df.itertuples()
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
            logger.info("Replace flag set, will compute and store ranks regardless")
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
                logger.debug("Prediction metadata does not match what is in configuration"
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
                logger.debug("At least one null in rankings in predictions table",
                              ", will compute and store ranks")
                return True
        logger.debug("No need to recompute prediction ranks")
        return False


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
            (np.Array) the generated prediction values
        """
        # Setting the Prediction object type - TrainPrediction or TestPrediction
        matrix_type = matrix_store.matrix_type

        if not self.replace:
            logger.info(
                f"Replace flag not set, looking for old predictions for model id {model_id} "
                f"on {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid}"
            )
            try:
                session = self.sessionmaker()
                existing_predictions = self._existing_predictions(
                    matrix_type.prediction_obj, session, model_id, matrix_store
                )
                logger.spam(f"Existing predictions length: {existing_predictions.count()}, Length of matrix: {len(matrix_store.index)}")
                if existing_predictions.count() == len(matrix_store.index):
                    logger.info(
                        f"Found old predictions for model id {model_id}, matrix {matrix_store.uuid}, returning saved versions"
                    )
                    return self._load_saved_predictions(existing_predictions, matrix_store)
            finally:
                session.close()

        model = self.load_model(model_id)
        if not model:
            raise ValueError(f"Model id {model_id} not found")
        logger.spam(f"Loaded model {model_id}")

        # Labels are popped from matrix (i.e. they are removed and returned)
        labels = matrix_store.labels

        # using a threading backend because the default loky backend doesn't
        # allow for nested parallelization (e.g., multiprocessing at triage level)
        with parallel_backend('threading'):
            predictions = model.predict_proba(
                matrix_store.matrix_with_sorted_columns(train_matrix_columns)
            )[:, 1]  # Returning only the scores for the label == 1


        logger.debug(
            f"Generated predictions for model {model_id} on {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid}"
        )
        if self.save_predictions:
            df = pd.DataFrame(data=None, columns=None, index=matrix_store.index)
            df['label_value'] = matrix_store.labels
            df['score'] = predictions


            logger.spam(f"Sorting predictions for model {model_id} using {self.rank_order}")

            if self.rank_order == 'best':
                df.sort_values(by=["score", "label_value"], inplace=True, ascending=[False,False], na_position='last')
            elif self.rank_order == 'worst':
                df.sort_values(by=["score", "label_value"], inplace=True, ascending=[False,True], na_position='first')
            elif self.rank_order == 'random':
                df['random'] = np.random.rand(len(df))
                df.sort_values(by=['score', 'random'], inplace=True, ascending=[False, False])
                df.drop('random', axis=1)
            else:
                raise ValueError(f"Rank order specified in condiguration file not recognized: {self.rank_order} ")

            df['rank_abs_no_ties'] = df['score'].rank(ascending=False, method='first')
            # uses the lowest rank in the group
            df['rank_abs_with_ties'] = df['score'].rank(ascending=False, method='min')
            # No gaps between groups (so it reaches 1.0). We are using rank_abs_no_ties so we can
            # respect that order (instead of using the mathematical formula,  as was done before)
            df['rank_pct_no_ties'] = df['rank_abs_no_ties'].rank(ascending=True, method='dense', pct=True)
            df['rank_pct_with_ties'] = df['score'].rank(ascending=False, method='dense', pct=True)

            df.reset_index(inplace=True)
            logger.debug(f"Predictions on {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid} from model {model_id} sorted using {self.rank_order}")

            logger.spam(
                f"Writing predictions for model {model_id} on {matrix_store.matrix_type.string_name}  matrix {matrix_store.uuid} to database"
            )

            self._write_predictions_to_db(
                model_id,
                matrix_store,
                df,
                misc_db_parameters,
                matrix_type.prediction_obj,
            )
            logger.debug(
                f"Wrote predictions for model {model_id} on  {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid} to database"
            )
        else:
            logger.notice(
                f"Predictions for model {model_id} on {matrix_store.matrix_type.string_name} matrix {matrix_store.uuid}  weren't written to the db because, because you asked not to do so"
            )
            logger.spam(f"Status of the save_predictions flag: {self.save_predictions}")

        self._write_metadata_to_db(
            model_id=model_id,
            matrix_uuid=matrix_store.uuid,
            matrix_type=matrix_type,
            random_seed=None,
        )

        return predictions
