import copy
import datetime
import importlib
import logging
import sys
from contextlib import contextmanager

import numpy as np
import pandas
from sklearn.model_selection import ParameterGrid
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Model, FeatureImportance
from triage.component.catwalk.exceptions import BaselineFeatureNotInMatrix

from .model_grouping import ModelGrouper
from .feature_importances import get_feature_importances
from .utils import (
    filename_friendly_hash,
    retrieve_model_id_from_hash,
    db_retry,
    save_db_objects,
)

NO_FEATURE_IMPORTANCE = (
    "Algorithm does not support a standard way" + " to calculate feature importance."
)


class ModelTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        project_path (string) path to project folder,
            under which to cache model pickles
        experiment_hash (string) foreign key to the model_metadata.experiments table
        model_storage_engine (catwalk.storage.ModelStorageEngine)
        db_engine (sqlalchemy.engine)
        replace (bool) whether or not to replace existing versions of models
    """

    def __init__(
        self,
        experiment_hash,
        model_storage_engine,
        db_engine,
        model_grouper=None,
        replace=True,
    ):
        self.experiment_hash = experiment_hash
        self.model_storage_engine = model_storage_engine
        self.model_grouper = model_grouper or ModelGrouper()
        self.db_engine = db_engine
        self.replace = replace

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    def unique_parameters(self, parameters):
        return {key: parameters[key] for key in parameters.keys() if key != "n_jobs"}

    def _model_hash(self, matrix_metadata, class_path, parameters):
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
            "className": class_path,
            "parameters": self.unique_parameters(parameters),
            "project_path": self.model_storage_engine.project_storage.project_path,
            "training_metadata": matrix_metadata,
        }
        logging.info("Creating model hash from unique data %s", unique)
        return filename_friendly_hash(unique)

    def _generate_model_configs(self, grid_config):
        """Flattens a model/parameter grid configuration into individually
        trainable model/parameter pairs

        Yields: (tuple) classpath and parameters
        """
        for class_path, parameter_config in grid_config.items():
            for parameters in ParameterGrid(parameter_config):
                yield class_path, parameters

    def _train(self, matrix_store, class_path, parameters):
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

        return instance.fit(matrix_store.design_matrix, matrix_store.labels)

    @db_retry
    def _save_feature_importances(self, model_id, feature_importances, feature_names):
        """Saves feature importances to the database.

        Deletes any existing feature importances for the given model_id.

        Args:
            model_id (int) The database id for the model
            feature_importances (numpy.ndarray, maybe). Calculated feature importances
                for the model
            feature_names (list) Feature names for the corresponding entries in feature_importances
        """
        self.db_engine.execute(
            "delete from train_results.feature_importances where model_id = %s",
            model_id,
        )
        db_objects = []
        if isinstance(feature_importances, np.ndarray):
            temp_df = pandas.DataFrame({"feature_importance": feature_importances})
            features_index = temp_df.index.tolist()
            rankings_abs = temp_df["feature_importance"].rank(
                method="dense", ascending=False
            )
            rankings_pct = temp_df["feature_importance"].rank(
                method="dense", ascending=False, pct=True
            )
            for feature_index, importance, rank_abs, rank_pct in zip(
                features_index, feature_importances, rankings_abs, rankings_pct
            ):
                db_objects.append(
                    FeatureImportance(
                        model_id=model_id,
                        feature_importance=round(float(importance), 10),
                        feature=feature_names[feature_index],
                        rank_abs=int(rank_abs),
                        rank_pct=round(float(rank_pct), 10),
                    )
                )
        # get_feature_importances was not able to find
        # feature importances
        else:
            db_objects.append(
                FeatureImportance(
                    model_id=model_id,
                    feature_importance=0,
                    feature=NO_FEATURE_IMPORTANCE,
                    rank_abs=0,
                    rank_pct=0,
                )
            )
        save_db_objects(self.db_engine, db_objects)

    @db_retry
    def _write_model_to_db(
        self,
        class_path,
        parameters,
        feature_names,
        model_hash,
        trained_model,
        model_group_id,
        model_size,
        misc_db_parameters,
    ):
        """Writes model and feature importance data to a database
        Will overwrite the data of any previous versions
        (any existing model that shares a hash)

        If the replace flag on the object is set, the existing version of the model
        will have its non-unique attributes (e.g. timestamps) updated,
        and feature importances fully replaced.

        If the replace flag on the object is not set, the existing model metadata
        and feature importances will be used.

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            feature_names (list) feature names in order given to model
            model_hash (string) a unique id for the model
            trained_model (object) a trained model object
            model_group_id (int) the unique id for the model group
            model_size (float) the size of the stored model in kB
            misc_db_parameters (dict) params to pass through to the database
        """
        model_id = retrieve_model_id_from_hash(self.db_engine, model_hash)
        if model_id and not self.replace:
            logging.info(
                "Metadata for model_id %s found in database. Reusing model metadata.",
                model_id,
            )
            return model_id
        else:
            model = Model(
                model_hash=model_hash,
                model_type=class_path,
                hyperparameters=parameters,
                model_group_id=model_group_id,
                built_by_experiment=self.experiment_hash,
                model_size=model_size,
                **misc_db_parameters,
            )
            session = self.sessionmaker()
            if model_id:
                logging.info(
                    "Found model id %s, updating non-unique attributes", model_id
                )
                model.model_id = model_id
                session.merge(model)
                session.commit()
            else:
                session.add(model)
                session.commit()
                model_id = model.model_id
                logging.info("Added new model id %s", model_id)
            session.close()

        logging.info("Saving feature importances for model_id %s", model_id)
        self._save_feature_importances(
            model_id, get_feature_importances(trained_model), feature_names
        )
        logging.info("Done saving feature importances for model_id %s", model_id)
        return model_id

    def _train_and_store_model(
        self, matrix_store, class_path, parameters, model_hash, misc_db_parameters
    ):
        """Train a model, cache it, and write metadata to a database

        Args:
            matrix_store(catwalk.storage.MatrixStore) a matrix and metadata
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            model_hash (string) a unique id for the model
            misc_db_parameters (dict) params to pass through to the database

        Returns: (int) a database id for the model
        """
        misc_db_parameters["run_time"] = datetime.datetime.now().isoformat()
        logging.info("Training and storing model for matrix uuid %s", matrix_store.uuid)
        trained_model = self._train(matrix_store, class_path, parameters)

        unique_parameters = self.unique_parameters(parameters)

        model_group_id = self.model_grouper.get_model_group_id(
            class_path, unique_parameters, matrix_store.metadata, self.db_engine
        )
        logging.info(
            "Trained model: hash %s, model group id %s ", model_hash, model_group_id
        )
        # Writing the model to storage, then getting its size in kilobytes.
        self.model_storage_engine.write(trained_model, model_hash)
        model_size = sys.getsizeof(trained_model) / (1024.0)

        logging.info("Cached model: %s", model_hash)
        model_id = self._write_model_to_db(
            class_path,
            unique_parameters,
            matrix_store.columns(include_label=False),
            model_hash,
            trained_model,
            model_group_id,
            model_size,
            misc_db_parameters,
        )
        logging.info("Wrote model to db: hash %s, got id %s", model_hash, model_id)
        return model_id

    @contextmanager
    def cache_models(self):
        """Caches each trained model in memory as it is written to storage.

        Must be used as a context manager.
        The cache is cleared when the context manager goes out of scope
        """
        with self.model_storage_engine.cache_models():
            yield

    def generate_trained_models(self, grid_config, misc_db_parameters, matrix_store):
        """Train and store configured models, yielding the ids one by one

        Args:
            grid_config (dict) of format {classpath: hyperparameter dicts}
                example: { 'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [1,10,100,1000,10000],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                } }
            misc_db_parameters (dict) params to pass through to the database
            matrix_store (catwalk.storage.MatrixStore) a matrix and metadata

        Yields: (int) model ids
        """
        for train_task in self.generate_train_tasks(
            grid_config, misc_db_parameters, matrix_store
        ):
            yield self.process_train_task(**train_task)

    def train_models(self, grid_config, misc_db_parameters, matrix_store):
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
            matrix_store(catwalk.storage.MatrixStore) a matrix and metadata

        Returns:
            (list) of model ids
        """
        return [
            model_id
            for model_id in self.generate_trained_models(
                grid_config, misc_db_parameters, matrix_store
            )
        ]

    def process_train_task(
        self, matrix_store, class_path, parameters, model_hash, misc_db_parameters
    ):
        """Trains and stores a model, or skips it and returns the existing id

        Args:
            matrix_store (catwalk.storage.MatrixStore) a matrix and metadata
            class_path (string): a full class path for the classifier
            parameters (dict): all hyperparameters to be passed to the classifier
            model_hash (string) a unique id for the model
            misc_db_parameters (dict) params to pass through to the database
        Returns: (int) model id
        """
        saved_model_id = retrieve_model_id_from_hash(self.db_engine, model_hash)
        if (
            not self.replace
            and self.model_storage_engine.exists(model_hash)
            and saved_model_id
        ):
            logging.info("Skipping %s/%s", class_path, parameters)
            return saved_model_id

        if self.replace:
            reason = "replace flag has been set"
        elif not self.model_storage_engine.exists(model_hash):
            reason = "model pickle not found in store"
        elif not saved_model_id:
            reason = "model metadata not found"

        logging.info(
            f"Training {class_path} with parameters {parameters}"
            f"(reason to train: {reason})"
        )
        try:
            model_id = self._train_and_store_model(
                matrix_store, class_path, parameters, model_hash, misc_db_parameters
            )
        except BaselineFeatureNotInMatrix:
            logging.warning(
                "Tried to train baseline model without required feature in matrix. Skipping."
            )
            model_id = None
        return model_id

    def generate_train_tasks(self, grid_config, misc_db_parameters, matrix_store=None):
        """Train and store configured models, yielding the ids one by one

        Args:
            grid_config (dict) of format {classpath: hyperparameter dicts}
                example: { 'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [1,10,100,1000,10000],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                } }
            misc_db_parameters (dict) params to pass through to the database

        Returns: (list) training task definitions, suitable for process_train_task kwargs
        """
        matrix_store = matrix_store or self.matrix_store
        misc_db_parameters = copy.deepcopy(misc_db_parameters)
        misc_db_parameters["batch_run_time"] = datetime.datetime.now().isoformat()
        misc_db_parameters["train_end_time"] = matrix_store.metadata["end_time"]
        misc_db_parameters["training_label_timespan"] = matrix_store.metadata[
            "label_timespan"
        ]
        misc_db_parameters["train_matrix_uuid"] = matrix_store.uuid

        tasks = []

        for class_path, parameters in self._generate_model_configs(grid_config):
            model_hash = self._model_hash(matrix_store.metadata, class_path, parameters)
            logging.info(
                f"Computed model hash for {class_path} "
                f"with parameters {parameters}: {model_hash}"
            )

            if any(task["model_hash"] == model_hash for task in tasks):
                logging.info(
                    "Skipping model_hash %s because another"
                    "equivalent one found in this batch."
                    "Classpath: %s -- Hyperparameters: %s",
                    model_hash,
                    class_path,
                    parameters,
                )
                continue
            tasks.append(
                {
                    "matrix_store": matrix_store,
                    "class_path": class_path,
                    "parameters": parameters,
                    "model_hash": model_hash,
                    "misc_db_parameters": misc_db_parameters,
                }
            )
        logging.info("Found %s unique model training tasks", len(tasks))
        return tasks
