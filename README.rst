===============================
Triage
===============================


.. image:: https://img.shields.io/travis/dssg/triage.svg
    :target: https://travis-ci.org/dssg/triage
    :alt: Build Status

.. image:: https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dssg/triage
    :alt: Code Coverage

.. image:: https://codeclimate.com/github/dssg/triage.png
    :target: https://codeclimate.com/github/dssg/triage
    :alt: Code Climate


Risk modeling and prediction


* Free software: MIT license
* Documentation: https://triage.readthedocs.io.


Basic Usage
--------

The simplest usage of triage is to use the Pipeline class. It requires:


* An experiment config. This contains time splitting, feature generation, grid search, and model scoring configuration. An up-to-date example is at example_experiment_config.yaml. This is passed in dict format to the Pipeline constructor
* a SQLAlchemy Postgres db engine. There is a convenience wrapper at triage.db.connect(), that reads from a local database.yaml file, but you supply an engine created however you like.
* Pick a model storage engine class from the `triage.storage` module. The model storage engines are wrappers around various ways to cache models, and are exclusively used for model caching inside triage components. Currently implemented are InMemoryModelStorageEngine (no caching), FSModelStorageEngine (on local or mounted disk), S3ModelStorageEngine (direct to s3).
* Pick a project path. This is used by the persistent storage engines.

Pipeline.run() is expected to be called after an ETL step. Given a properly configured experiment config, it will handle label generation, training/test splits, feature generation (using collate), training, testing, and scoring. The model metadata, feature importances, predictions, and model scores are saved to the `results` schema in the given database. The trained model pickles are saved to the storage engine passed into the Pipeline constructor, along with metadata files.


.. code-block::

    from triage.db import connect
    from triage.storage import FSModelStorageEngine
    from triage.pipeline import Pipeline


        with open('example_experiment_config.yaml') as f:
            experiment_config = yaml.load(f)
        pipeline = Pipeline(
            config=experiment_config,
            db_engine=connect(),
            model_storage_class=FSModelStorageEngine,
            project_path='/path/to/cached_models/test_project'
        )

        pipeline.run()


Advanced Usage
--------

The current method of changing behavior beyond what can be controlled in the experiment config file is to use the inner components (e.g. FeatureGenerator, ModelTrainer, etc). You can copy the Pipeline class at triage/pipeline.py and modify the contents. Support will be added in the future for subclassing the pipeline and/or passing in subclassed components like an alternate label generator.


Features
--------

* TODO


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

