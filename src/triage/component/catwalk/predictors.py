import logging
import math
from datetime import datetime

import numpy
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Model, Prediction

from .utils import db_retry, save_db_objects


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
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)
        self.replace = replace

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
    def _existing_predictions(self, session, model_id, matrix_store):
        return session.query(Prediction)\
            .filter_by(model_id=model_id)\
            .filter(Prediction.as_of_date.in_(self._as_of_dates(matrix_store)))

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

        """
        session = self.sessionmaker()
        self._existing_predictions(session, model_id, matrix_store)\
            .delete(synchronize_session=False)
        session.expire_all()
        test_label_timespan = matrix_store.metadata['label_timespan']
        logging.warning(test_label_timespan)
        session.commit()
        session.close()

        db_objects_generator = None
        if 'as_of_date' in matrix_store.matrix.index.names:
            logging.info('as_of_date found as part of matrix index, using '
                         'index for table as_of_dates')

            db_objects_generator = (
                Prediction(
                    model_id=int(model_id),
                    entity_id=int(index[0]),
                    as_of_date=index[1],
                    score=float(score),
                    label_value=int(label) if not math.isnan(label) else None,
                    matrix_uuid=matrix_store.uuid,
                    test_label_timespan=test_label_timespan,
                    **misc_db_parameters
                )
                for index, score, label in zip(
                    matrix_store.matrix.index,
                    predictions,
                    labels
                )
            )
        else:
            logging.info('as_of_date not found as part of matrix index, using '
                         'matrix metadata end_time as as_of_date')
            db_objects_generator = (
                Prediction(
                    model_id=int(model_id),
                    entity_id=int(entity_id),
                    as_of_date=matrix_store.metadata['end_time'],
                    score=round(float(score), 10),
                    label_value=int(label) if not math.isnan(label) else None,
                    matrix_uuid=matrix_store.uuid,
                    test_label_timespan=test_label_timespan,
                    **misc_db_parameters
                )
                for entity_id, score, label in zip(
                    matrix_store.matrix.index,
                    predictions,
                    labels,
                )
            )
        logging.info(
            'Beginning COPY of Prediction database objects for model %s, matrix %s',
            model_id,
            matrix_store.uuid
        )
        save_db_objects(self.db_engine, db_objects_generator)
        logging.info(
            'Completed COPY of Prediction database objects for model %s, matrix %s',
            model_id,
            matrix_store.uuid
        )
        logging.info(
            'Beginning ranking of new Predictions for model %s, matrix %s',
            model_id,
            matrix_store.uuid
        )
        with self.db_engine.begin() as conn:
            conn.execute('''
                with ranks as (
                    select
                        entity_id,
                        as_of_date,
                        dense_rank() over (
                            partition by as_of_date order by score desc
                        ) as abs_rank,
                        percent_rank() over (
                            partition by as_of_date order by score desc
                        ) as pct_rank
                        from results.predictions
                        where model_id = %(model_id)s and matrix_uuid = %(uuid)s
                )
                update results.predictions as p
                set
                    rank_abs = r.abs_rank,
                    rank_pct = r.pct_rank
                from ranks as r
                where
                    p.model_id = %(model_id)s
                    and p.matrix_uuid = %(uuid)s
                    and p.entity_id = r.entity_id
                    and p.as_of_date = r.as_of_date
            ''', model_id=model_id, uuid=matrix_store.uuid)
        logging.info(
            'Completed ranking of new Predictions for model %s, matrix %s',
            model_id,
            matrix_store.uuid
        )

    def predict(self, model_id, matrix_store, misc_db_parameters, train_matrix_columns):
        """Generate predictions and store them in the database

        Args:
            model_id (int) the id of the trained model to predict based off of
            matrix_store (catwalk.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata
            misc_db_parameters (dict): attributes and values to add to each
                Prediction object in the results schema
            train_matrix_columns (list): The order of columns that the model
                was trained on

        Returns:
            (numpy.Array) the generated prediction values
        """
        session = self.sessionmaker()
        if not self.replace:
            logging.info(
                'replace flag not set for model id %s, matrix %s, looking for old predictions',
                model_id,
                matrix_store.uuid
            )
            existing_predictions = self._existing_predictions(
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
            misc_db_parameters
        )
        logging.info(
            'Wrote predictions for model %s, matrix %s to database',
            model_id,
            matrix_store.uuid
        )
        return predictions_proba[:, 1]
