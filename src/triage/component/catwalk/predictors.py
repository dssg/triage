import csv
import logging
import math
import tempfile
from datetime import datetime

import numpy
import pandas
import postgres_copy
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Model

from .utils import db_retry


class ModelNotFoundError(ValueError):
    pass


class Predictor(object):
    expected_matrix_ts_format = '%Y-%m-%d %H:%M:%S'

    def __init__(
        self,
        project_path,
        model_storage_engine,
        db_engine,
        replace=True
    ):
        """Encapsulates the task of generating predictions on an arbitrary
        dataset and storing the results

        Args:
            project_path (string) the path under which to store project data
            model_storage_engine (catwalk.storage.ModelStorageEngine)
            db_engine (sqlalchemy.engine)

        """
        self.project_path = project_path
        self.model_storage_engine = model_storage_engine
        self.db_engine = db_engine
        self.replace = replace

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    @db_retry
    def _retrieve_model_hash(self, model_id):
        """Retrieves the model hash associated with a given model id

        Args:
            model_id (int) The id of a given model in the database

        Returns: (str) the stored hash of the model
        """
        try:
            session = self.sessionmaker()
            model_hash = session.query(Model).get(model_id).model_hash
        finally:
            session.close()
        return model_hash

    @db_retry
    def load_model(self, model_id):
        """Downloads the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database

        Returns:
            A python object which implements .predict()
        """

        model_hash = self._retrieve_model_hash(model_id)
        logging.info('Checking for model_hash %s in store', model_hash)
        model_store = self.model_storage_engine.get_store(model_hash)
        if model_store.exists():
            return model_store.load()

    @db_retry
    def delete_model(self, model_id):
        """Deletes the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database
        """
        model_hash = self._retrieve_model_hash(model_id)
        model_store = self.model_storage_engine.get_store(model_hash)
        model_store.delete()

    @db_retry
    def _existing_predictions(self, Prediction_obj, session, model_id, matrix_store):
        return session.query(Prediction_obj)\
            .filter_by(model_id=model_id)\
            .filter(Prediction_obj.as_of_date.in_(self._as_of_dates(matrix_store)))

    def _as_of_dates(self, matrix_store):
        matrix = matrix_store.matrix
        if 'as_of_date' in matrix.index.names:
            return matrix.index.levels[
                matrix.index.names.index('as_of_date')
            ].tolist()
        else:
            return [matrix_store.metadata['end_time']]

    @db_retry
    def _load_saved_predictions(self, existing_predictions, matrix_store):
        index = matrix_store.matrix.index
        score_lookup = {}
        for prediction in existing_predictions:
            score_lookup[(
                prediction.entity_id,
                prediction.as_of_date.date().isoformat()
            )] = prediction.score
        if 'as_of_date' in index.names:
            score_iterator = (
                score_lookup[(
                    entity_id,
                    datetime.strptime(dt, self.expected_matrix_ts_format).date().isoformat()
                )]
                for (entity_id, dt) in index
            )
        else:
            as_of_date = matrix_store.metadata['end_time'].date().isoformat()
            score_iterator = (score_lookup[(row, as_of_date)] for row in index)
        return numpy.fromiter(score_iterator, float)

    @db_retry
    def _write_to_db(
        self,
        model_id,
        matrix_store,
        predictions,
        labels,
        misc_db_parameters,
        Prediction_obj
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
        session = self.sessionmaker()
        self._existing_predictions(Prediction_obj, session, model_id, matrix_store)\
            .delete(synchronize_session=False)
        session.expire_all()
        db_objects = []
        test_label_timespan = matrix_store.metadata['label_timespan']
        logging.warning(test_label_timespan)

        if 'as_of_date' in matrix_store.matrix.index.names:
            logging.info('as_of_date found as part of matrix index, using '
                         'index for table as_of_dates')
            session.commit()
            session.close()
            with tempfile.TemporaryFile(mode='w+') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                for index, score, label in zip(
                    matrix_store.matrix.index,
                    predictions,
                    labels
                ):
                    entity_id, as_of_date = index
                    prediction = Prediction_obj(
                        model_id=int(model_id),
                        entity_id=int(entity_id),
                        as_of_date=as_of_date,
                        score=float(score),
                        label_value=int(label) if not math.isnan(label) else None,
                        matrix_uuid=matrix_store.uuid,
                        test_label_timespan=test_label_timespan,
                        **misc_db_parameters
                    )
                    writer.writerow([
                        prediction.model_id,
                        prediction.entity_id,
                        prediction.as_of_date,
                        prediction.score,
                        prediction.label_value,
                        prediction.rank_abs,
                        prediction.rank_pct,
                        prediction.matrix_uuid,
                        prediction.test_label_timespan
                    ])
                f.seek(0)
                postgres_copy.copy_from(f, Prediction_obj, self.db_engine, format='csv')
        else:
            logging.info('as_of_date not found as part of matrix index, using '
                         'matrix metadata end_time as as_of_date')
            temp_df = pandas.DataFrame({'score': predictions})
            rankings_abs = temp_df['score'].rank(method='dense', ascending=False)
            rankings_pct = temp_df['score'].rank(method='dense', ascending=False, pct=True)
            for entity_id, score, label, rank_abs, rank_pct in zip(
                matrix_store.matrix.index,
                predictions,
                labels,
                rankings_abs,
                rankings_pct
            ):
                db_objects.append(Prediction_obj(
                    model_id=int(model_id),
                    entity_id=int(entity_id),
                    as_of_date=matrix_store.metadata['end_time'],
                    score=round(float(score), 10),
                    label_value=int(label) if not math.isnan(label) else None,
                    rank_abs=int(rank_abs),
                    rank_pct=round(float(rank_pct), 10),
                    matrix_uuid=matrix_store.uuid,
                    test_label_timespan=test_label_timespan,
                    **misc_db_parameters
                ))

            session.bulk_save_objects(db_objects)
            session.commit()
            session.close()

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
        #Setting the Prediction object type - TrainPrediction or TestPrediction
        prediction_obj = matrix_store.matrix_type.prediction_obj

        session = self.sessionmaker()
        if not self.replace:
            logging.info(
                'replace flag not set for model id %s, matrix %s, looking for old predictions',
                model_id,
                matrix_store.uuid
            )
            existing_predictions = self._existing_predictions(
                prediction_obj,
                session,
                model_id,
                matrix_store
            )
            index = matrix_store.matrix.index
            if existing_predictions.count() == len(index):
                logging.info(
                    'Found predictions for model id %s, matrix %s, returning saved versions',
                    model_id,
                    matrix_store.uuid
                )
                return self._load_saved_predictions(
                    existing_predictions,
                    matrix_store
                )

        model = self.load_model(model_id)
        logging.info('Loaded model %s', model_id)
        if not model:
            raise ModelNotFoundError('Model id {} not found'.format(model_id))

        # Labels are popped from matrix (IE, they are removed and returned)
        labels = matrix_store.labels()

        predictions_proba = model.predict_proba(
            matrix_store.matrix_with_sorted_columns(train_matrix_columns)
        )
        logging.info('Generated predictions for model %s, matrix %s', model_id, matrix_store.uuid)

        self._write_to_db(
            model_id,
            matrix_store,
            predictions_proba[:, 1],
            labels,
            misc_db_parameters,
            prediction_obj
        )
        logging.info(
            'Wrote predictions for model %s, matrix %s to database',
            model_id,
            matrix_store.uuid
        )
        return predictions_proba[:, 1]
