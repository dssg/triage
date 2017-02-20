from sklearn.grid_search import ParameterGrid
from sqlalchemy.orm import sessionmaker
from .db import Model, FeatureImportance
import importlib
import json
import logging
import datetime
import copy
import pandas
from triage.utils import filename_friendly_hash


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


class ModelTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        project_path (string) path to project folder,
            under which to cache model pickles
        model_storage_engine (triage.storage.ModelStorageEngine)
        matrix_storage (triage.storage.MatrixStore)
        db_engine (sqlalchemy.engine)
    """
    def __init__(
        self,
        project_path,
        model_storage_engine,
        matrix_store,
        db_engine
    ):
        self.project_path = project_path
        self.model_storage_engine = model_storage_engine
        self.matrix_store = matrix_store
        self.db_engine = db_engine
        self.sessionmaker = sessionmaker(bind=self.db_engine)

    def _model_hash(self, class_path, parameters):
        """Generates a unique identifier for a trained model
        based on attributes of the model that together define
        equivalence; in other words, if we train a second model with these
        same attributes there would be no reason to keep the old one)

        Args:
        class_path (string): a full class path for the classifier
        parameters (dict): all hyperparameters to be passed to the classifier

        Returns: (string) a unique identifier
        """
        unique = {
            'className': class_path,
            'parameters': parameters,
            'project_path': self.project_path,
            'training_metadata': self.matrix_store.metadata
        }
        return filename_friendly_hash(unique)

    def _generate_model_configs(self, grid_config):
        """Flattens a model/parameter grid configuration into individually
        trainable model/parameter pairs

        Yields: (tuple) classpath and parameters
        """
        for class_path, parameter_config in grid_config.items():
            for parameters in ParameterGrid(parameter_config):
                yield class_path, parameters

    def _train(self, class_path, parameters):
        """Fit a model to a training set. Works on any modeling class that
        is available in this package's environment and implements .fit

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor

        Returns:
            tuple of (fitted model, list of column names without label)
        """
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**parameters)
        y = self.matrix_store.labels()

        return instance.fit(self.matrix_store.matrix, y), self.matrix_store.matrix.columns

    def _write_model_to_db(
        self,
        class_path,
        parameters,
        feature_names,
        model_hash,
        trained_model,
        model_group_id,
        misc_db_parameters
    ):
        """Writes model and feature importance data to a database
        Will overwrite the data of any previous versions
        (any existing model that shares a hash)

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            feature_names (list) feature names in order given to model
            model_hash (string) a unique id for the model
            trained_model (object) a trained model object
            misc_db_parameters (dict) params to pass through to the database
        """
        session = self.sessionmaker()
        existing_model = session.query(Model)\
            .filter_by(model_hash=model_hash)\
            .one_or_none()
        if existing_model:
            logging.warning('deleting existing model %s', existing_model.model_id)
            existing_model.delete(session)
            session.commit()

        model = Model(
            model_hash=model_hash,
            model_type=class_path,
            model_parameters=parameters,
            model_group_id=model_group_id,
            **misc_db_parameters
        )
        session.add(model)

        feature_importance = get_feature_importances(trained_model)
        temp_df = pandas.DataFrame({'feature_importance': feature_importance})
        features_index = temp_df.index.tolist()
        rankings_abs = temp_df['feature_importance'].rank(method='dense', ascending=False)
        rankings_pct = temp_df['feature_importance'].rank(method='dense', ascending=False, pct=True)
        for feature_index, importance, rank_abs, rank_pct in zip(
            features_index,
            feature_importance,
            rankings_abs,
            rankings_pct):
            feature_importance = FeatureImportance(
                model=model,
                feature_importance=importance,
                feature=feature_names[feature_index],
                rank_abs=int(rank_abs),
                rank_pct=float(rank_pct)
            )
            session.add(feature_importance)
        session.commit()
        return model.model_id

    def _train_and_store_model(
        self,
        class_path,
        parameters,
        model_hash,
        model_store,
        misc_db_parameters
    ):
        """Train a model, cache it, and write metadata to a database

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            model_hash (string) a unique id for the model
            model_store (triage.storage.Store) the place in which to store the model
            misc_db_parameters (dict) params to pass through to the database

        Returns: (int) a database id for the model
        """
        misc_db_parameters['run_time'] = datetime.datetime.now().isoformat()
        trained_model, feature_names = self._train(
            class_path,
            parameters,
        )
        
        model_group_id = self._get_model_group_id(
             class_path,
             parameters,
             self.matrix_store.metadata['prediction_window'],
             self.matrix_store.metadata['feature_names']
        ) 
        logging.debug('Trained model')
        model_store.write(trained_model)
        logging.debug('Cached model')
        model_id = self._write_model_to_db(
            class_path,
            parameters,
            feature_names,
            model_hash,
            trained_model,
            model_group_id,
            misc_db_parameters
        )
        logging.info('Wrote model to db')
        return model_id

    def _get_model_group_id(
        self,
        class_path,
        parameters,
        prediction_window,
        feature_names
    ):
        """
        Returns model group id using store procedure 'get_model_group_id' which will 
        return the same value for models with the same class_path, parameters, 
        prediction_window and features

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            prediction_window (string) The prediction window used for generating the labels
                                       stored in metadata
           features_names (list) Features used for train/test

        Returns: (int) a database id for the model group id
        """
        db_conn = self.db_engine.raw_connection()
        cur = db_conn.cursor()
        cur.execute( "SELECT EXISTS ( "
                     "       SELECT * "
                     "       FROM pg_catalog.pg_proc "
                     "       WHERE proname = 'get_model_group_id' ) ")
        condition = cur.fetchone()
        confition = condition[0]

        if condition:
            query = ("SELECT get_model_group_id( "
                     "            '{class_path}'::TEXT, "
                     "            '{parameters}'::JSONB, "
                     "            '{prediction_window}'::TEXT, "
                     "             ARRAY{feature_names}::TEXT [] )"
                     .format(class_path=class_path,
                             parameters=json.dumps(parameters),
                             prediction_window=prediction_window,
                             feature_names=feature_names))
            cur.execute(query)
            db_conn.commit()
            model_group_id = cur.fetchone()
            model_group_id = model_group_id[0]

        else:
            logging.info("Could not found store procedure public.pivot_table")
            model_group_id = None
        db_conn.close()

        logging.debug('Model_group_id = {}'.format(model_group_id))
        return model_group_id         

    def train_models(
        self,
        grid_config,
        misc_db_parameters,
        replace=False
    ):
        """Train and store configured models

        Args:
            grid_config (dict) of format {classpath: hyperparameter dicts}
                example: { 'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [1,10,100,1000,10000],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                } }
            misc_db_parameters (dict) params to pass through to the database
            replace (optional, False): whether to replace already cached models

        Returns:
            (list) of model ids
        """
        model_ids = []
        misc_db_parameters = copy.deepcopy(misc_db_parameters)
        misc_db_parameters['batch_run_time'] = datetime.datetime.now().isoformat()
        for class_path, parameters in self._generate_model_configs(grid_config):
            model_hash = self._model_hash(class_path, parameters)
            model_store = self.model_storage_engine.get_store(model_hash)
            if replace or not model_store.exists():
                logging.info('Training %s/%s', class_path, parameters)
                model_id = self._train_and_store_model(
                    class_path,
                    parameters,
                    model_hash,
                    model_store,
                    misc_db_parameters
                )
                model_ids.append(model_id)
            else:
                logging.info('Skipping %s/%s', class_path, parameters)
                session = self.sessionmaker()
                saved = session.query(Model)\
                    .filter_by(model_hash=model_hash)\
                    .one_or_none()
                session.close()
                if not saved:
                    logging.warning(
                        '%s/%s cache found but metadata not found, retraining',
                        class_path,
                        parameters
                    )
                    model_id = self._train_and_store_model(
                        class_path,
                        parameters,
                        model_hash,
                        model_store,
                        misc_db_parameters
                    )
                else:
                    model_id = saved.model_id
                model_ids.append(model_id)

        return model_ids
