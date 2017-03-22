from .db import Model, Prediction
from sqlalchemy.orm import sessionmaker
import pandas
import logging
import math


class ModelNotFoundError(ValueError):
    pass


class Predictor(object):
    def __init__(self, project_path, model_storage_engine, db_engine):
        """Encapsulates the task of generating predictions on an arbitrary
        dataset and storing the results

        Args:
            project_path (string) the path under which to store project data
            model_storage_engine (triage.storage.ModelStorageEngine)
            db_engine (sqlalchemy.engine)
        """
        self.project_path = project_path
        self.model_storage_engine = model_storage_engine
        self.db_engine = db_engine
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)

    def load_model(self, model_id):
        """Downloads the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database

        Returns:
            A python object which implements .predict()
        """
        model_hash = self.sessionmaker().query(Model).get(model_id).model_hash
        model_store = self.model_storage_engine.get_store(model_hash)
        if model_store.exists():
            return model_store.load()

    def delete_model(self, model_id):
        """Deletes the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database
        """
        model_hash = self.sessionmaker().query(Model).get(model_id).model_hash
        model_store = self.model_storage_engine.get_store(model_hash)
        model_store.delete()

    def _write_to_db(
        self,
        model_id,
        matrix_end_time,
        matrix,
        predictions,
        labels,
        misc_db_parameters,
    ):
        """Writes given predictions to database

        entity_ids, predictions, labels are expected to be in the same order

        Args:
            model_id (int) the id of the model associated with the given predictions
            as_of_date (datetime.date) the date the predictions were made 'as of'
            entity_ids (iterable) entity ids that predictions were made on
            predictions (iterable) predicted values
            labels (iterable) labels of prediction set (int) the id of the model to predict based off of
        """
        if 'as_of_date' in matrix.index.names:
            as_of_dates = matrix.index.levels[
                matrix.index.names.index('as_of_date')
            ].tolist()
        else:
            as_of_dates = [matrix_end_time]
        session = self.sessionmaker()
        session.query(Prediction)\
            .filter_by(model_id=model_id)\
            .filter(Prediction.as_of_date.in_(as_of_dates))\
            .delete(synchronize_session=False)
        session.expire_all()
        db_objects = []

        if 'as_of_date' in matrix.index.names:
            for index, score, label in zip(
                matrix.index,
                predictions,
                labels
            ):
                entity_id, as_of_date = index
                db_objects.append(Prediction(
                    model_id=int(model_id),
                    entity_id=int(entity_id),
                    as_of_date=as_of_date,
                    score=float(score),
                    label_value=int(label) if not math.isnan(label) else None,
                    **misc_db_parameters
                ))
        else:
            temp_df = pandas.DataFrame({'score': predictions})
            rankings_abs = temp_df['score'].rank(method='dense', ascending=False)
            rankings_pct = temp_df['score'].rank(method='dense', ascending=False, pct=True)
            for entity_id, score, label, rank_abs, rank_pct in zip(
                matrix.index,
                predictions,
                labels,
                rankings_abs,
                rankings_pct
            ):
                db_objects.append(Prediction(
                    model_id=int(model_id),
                    entity_id=int(entity_id),
                    as_of_date=matrix_end_time,
                    score=float(score),
                    label_value=int(label) if not math.isnan(label) else None,
                    rank_abs=int(rank_abs),
                    rank_pct=float(rank_pct),
                    **misc_db_parameters
                ))

        session.bulk_save_objects(db_objects)
        session.commit()

    def predict(self, model_id, matrix_store, misc_db_parameters):
        """Generate predictions and store them in the database

        Args:
            model_id (int) the id of the trained model to predict based off of
            matrix_store (triage.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata

        Returns:
            (numpy.Array) the generated prediction values
        """
        model = self.load_model(model_id)
        if not model:
            raise ModelNotFoundError('Model id {} not found'.format(model_id))
        labels = matrix_store.labels()
        predictions = model.predict(matrix_store.matrix)
        predictions_proba = model.predict_proba(matrix_store.matrix)
        self._write_to_db(
            model_id,
            matrix_store.metadata['end_time'],
            matrix_store.matrix,
            predictions_proba[:,1],
            labels,
            misc_db_parameters
        )
        return predictions, predictions_proba[:,1]
