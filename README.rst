===============================
Triage
===============================


.. image:: https://img.shields.io/travis/dssg/triage.svg
        :target: https://travis-ci.org/dssg/triage

.. image:: https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg
	 :target: https://codecov.io/gh/dssg/triage
	 :alt: Code Coverage


Risk modeling and prediction


* Free software: MIT license
* Documentation: https://triage.readthedocs.io.


Features
--------

* TODO

Usage
--------

::
    from triage.storage import S3ModelStorageEngine
    from triage.model_trainers import SimpleModelTrainer
    from triage.predictors import Predictor

    db_engine = ...
    s3_conn = ...
    project_path = 'econ-dev/inspections'
    model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)

    trainer = SimpleModelTrainer(
        project_path=project_path,
        model_storage_engine,
        db_engine=db_engine
    )
    predictor = Predictor(
        project_path=project_path,
        model_storage_engine=model_storage_engine,
        db_engine=db_engine
    )

    model_ids = trainer.train_models(
        training_set_path=...,
        training_metadata_path=...,
        model_config=...
    )

    for model_id in model_ids:
        predictions = predictor.predict(
            model_id,
            test_matrix_path=...,
            test_metadata_path=...
        )


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

