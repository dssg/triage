from sklearn.grid_search import ParameterGrid
from sqlalchemy.orm import sessionmaker
from .db import Model, FeatureImportance
from .utils import split_s3_path, upload_object_to_key, key_exists
import importlib
import json
import logging
import pandas
import yaml


def get_feature_importances(model):
    """
    Get feature importances (from scikit-learn) of trained model.

    Args:
        model: Trained model

    Returns:
        Feature importances, or failing that, None
    """

    try:
        return model.feature_importances_
    except:
        pass
    try:
        # Must be 1D for feature importance plot
        if len(model.coef_) <= 1:
            return model.coef_[0]
        else:
            return model.coef_
    except:
        pass
    return None


class SimpleModelTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        training_set_path (string): filepath to (hdf5) training set
        training_metadata_path (string): filepath to (yaml) training metadata
        model_config (dict) of format {classpath: hyperparameter dicts}
            example: { 'sklearn.ensemble.RandomForestClassifier': {
                'n_estimators': [1,10,100,1000,10000],
                'max_depth': [1,5,10,20,50,100],
                'max_features': ['sqrt','log2'],
                'min_samples_split': [2,5,10]
            } }
        project_path (string) path to project folder on s3,
            under which to cache model pickles
        s3_conn (boto3.s3.connection)
        db_engine (sqlalchemy.engine)
    """
    def __init__(
        self,
        training_set_path,
        training_metadata_path,
        model_config,
        project_path,
        s3_conn,
        db_engine=None
    ):
        self.training_set_path = training_set_path
        self.training_metadata_path = training_metadata_path
        self.model_config = model_config
        self.project_path = project_path
        self.s3_conn = s3_conn
        self.db_engine = db_engine
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)

    def _model_id(self, class_path, parameters):
        """Generates a unique identifier for a trained model
        based on attributes of the model that together define
        equivalence; in other words, if we train a second model with these
        same attributes there would be no reason to keep the old one)

        Args:
        class_path (string): a full class path for the classifier
        parameters (dict): all hyperparameters to be passed to the classifier

        Returns: (string) a unique identifier
        """
        with open(self.training_metadata_path) as f:
            training_metadata = yaml.load(f)

        unique = {
            'className': class_path,
            'parameters': parameters,
            'project_path': self.project_path,
            'training_metadata': training_metadata
        }
        return hex(hash(json.dumps(unique, sort_keys=True)))

    def _generate_model_configs(self):
        """Flattens a model/parameter grid configuration into individually
        trainable model/parameter pairs

        Yields: (tuple) classpath and parameters
        """
        for class_path, parameter_config in self.model_config.items():
            for parameters in ParameterGrid(parameter_config):
                yield class_path, parameters

    def _output_cache_key(self, model_id):
        """Generates an s3 key for a given model_id

        Args:
            model_id (string) a unique model id

        Returns:
            (boto3.s3.Object) an s3 key, which may or may not have contents
        """
        bucket_name, prefix = split_s3_path(self.project_path)
        path = '/'.join([prefix, 'trained_models', model_id])
        return self.s3_conn.Object(bucket_name, path)

    def _train(self, class_path, parameters, label_name, train_matrix):
        """Fit a model to a training set. Works on any modeling class that
        is available in this package's environment and implements .fit

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            label_name (string) the name of the label column in the matrix
            train_matrix (pandas.DataFrame) the training matrix including label

        Returns:
            tuple of (fitted model, list of column names without label)
        """
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**parameters)
        y = train_matrix.pop(label_name)

        return instance.fit(train_matrix, y), train_matrix.columns

    def _write_model_to_db(
        self,
        class_path,
        parameters,
        feature_names,
        model_id,
        trained_model
    ):
        """Writes model and feature importance data to a database

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            feature_names (list) feature names in order given to model
            model_id (string) a unique id for the model
            trained_model a trained model
        """
        session = self.sessionmaker()
        model = Model(
            unique_identifier=model_id,
            model_type=class_path,
            model_parameters=parameters
        )
        session.add(model)

        for feature_index, importance in \
                enumerate(get_feature_importances(trained_model)):
            feature_importance = FeatureImportance(
                model=model,
                feature_importance=importance,
                feature=feature_names[feature_index],
            )
            session.add(feature_importance)
        session.commit()

    def _get_train_matrix_and_metadata(self):
        """Retrieve a training matrix in hdf format and
        training data in yaml format

        Returns: (tuple) training matrix, training metadata
        """
        matrix = pandas.read_hdf(self.training_set_path)
        with open(self.training_metadata_path) as f:
            metadata = yaml.load(f)
        return matrix, metadata

    def _train_and_store_model(
        self,
        class_path,
        parameters,
        model_id,
        cache_key
    ):
        """Train a model, cache it in s3, and write metadata to a database

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            model_id (string) a unique id for the model
            cache_key (boto3.s3.Object) the s3 key in which to store the model
        """
        matrix, metadata = self._get_train_matrix_and_metadata()
        trained_model, feature_names = self._train(
            class_path,
            parameters,
            metadata['label_name'],
            matrix
        )
        logging.info('Trained model')
        upload_object_to_key(trained_model, cache_key)
        logging.info('Cached model')
        self._write_model_to_db(
            class_path,
            parameters,
            feature_names,
            model_id,
            trained_model
        )
        logging.info('Wrote model to db')

    def train_models(self, replace=False):
        """Train and store configured models

        Args:
            replace (optional, False): whether to replace already cached models

        Returns:
            (list) of s3 cache keys where the trained models can be accessed
        """
        cache_keys = []
        for class_path, parameters in self._generate_model_configs():
            model_id = self._model_id(class_path, parameters)
            cache_key = self._output_cache_key(model_id)
            if replace or not key_exists(cache_key):
                logging.info('Training %s/%s', class_path, parameters)
                self._train_and_store_model(
                    class_path,
                    parameters,
                    model_id,
                    cache_key
                )
                cache_keys.append(cache_key)
            else:
                logging.info('Skipping %s/%s', class_path, parameters)

        return cache_keys
