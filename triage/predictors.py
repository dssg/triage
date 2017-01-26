from .db import Model, Prediction
from .utils import model_cache_key, download_object, get_matrix_and_metadata
from sqlalchemy.orm import sessionmaker


class Predictor(object):
    def __init__(self, project_path, s3_conn, db_engine):
        """Encapsulates the task of generating predictions on an arbitrary
        dataset and storing the results

        Args:
            project_path (string) a desired fs/s3 project path
            s3_conn (boto3.s3.connection)
            db_engine (sqlalchemy.engine)
        """
        self.project_path = project_path
        self.db_engine = db_engine
        self.s3_conn = s3_conn
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)

    def _load_model(self, model_id):
        """Downloads the cached model associated with a given model id

        Args:
            model_id (int) The id of a given model in the database

        Returns:
            A python object which implements .predict()
        """
        cache_key = model_cache_key(
            self.project_path,
            self.sessionmaker().query(Model).get(model_id).model_hash,
            self.s3_conn
        )
        return download_object(cache_key)

    def _write_to_db(self, model_id, as_of_date, entity_ids, predictions, labels):
        """Writes given predictions to database

        entity_ids, predictions, labels are expected to be in the same order

        Args:
            model_id (int) the id of the model associated with the given predictions
            as_of_date (datetime.date) the date the predictions were made 'as of'
            entity_ids (iterable) entity ids that predictions were made on
            predictions (iterable) predicted values
            labels (iterable) labels of prediction set (int) the id of the model to predict based off of
            dataset_path (string)
        """
        session = self.sessionmaker()
        for entity_id, score, label in zip(entity_ids, predictions, labels):
            prediction = Prediction(
                model_id=model_id,
                entity_id=int(entity_id),
                as_of_date=as_of_date,
                entity_score=score,
                label_value=int(label)
            )
            session.add(prediction)
        session.commit()

    def predict(self, model_id, dataset_path, metadata_path):
        """Generate predictions and store them in the database

        Args:
            model_id (int) the id of the trained model to predict based off of
            dataset_path (string) filepath of the dataset to predict
                Expected to be in a form consistent with metta-data,
                and also for the entity ids to be the only index column
            metadata_path (string) path to a yaml file describing the dataset,
                in a form consistent with metta-data

        Returns:
            (numpy.Array) the generated prediction values
        """
        dataset, metadata = get_matrix_and_metadata(dataset_path, metadata_path)
        label_name = metadata['label_name']
        model = self._load_model(model_id)
        labels = dataset.pop(label_name)
        as_of_date = metadata['end_time']
        predictions = model.predict(dataset)
        self._write_to_db(model_id, as_of_date, dataset.index, predictions, labels)
        return predictions
