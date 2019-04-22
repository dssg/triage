=======
Catwalk
=======

Training, testing, and evaluating machine learning classifier models

At the core of many predictive analytics applications is the need to train classifiers on large set of design matrices, test and temporally cross-validate them, and generate evaluation metrics about them.

Python's scikit-learn package provides much of this functionality, but it is not trivial to design large experiments with it in a persistable way. Catwalk builds upon the functionality offered by scikit-learn by implementing:

- Saving of modeling results and metadata in a `Postgres database <https://github.com/dssg/results-schema>`_ for later analysis
- Exposure of computationally-intensive tasks as discrete workloads that can be used with different parallelization solutions (e.g. multiprocessing, Celery)
- Different model persistence strategies such as on-filesystem or Amazon S3, that can be easily switched between
- Hashing classifier model configuration to only retrain a model if necessary.
- Various best practices in areas like input scaling for different classifier types and feature importance
- Common scikit-learn model evaluation metrics as well as the ability to bundle custom evaluation metrics
- Custom model wrappers for classifiers
- 'Baseline' classes that generate classifications or predictions based on pre-determined rules, to be used for evaluating predictive models against simple hueristics
