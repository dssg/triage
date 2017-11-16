# The Architect 

Plan, design, and build train and test matrices

[![Build Status](https://travis-ci.org/dssg/architect.svg?branch=master)](https://travis-ci.org/dssg/architect)
[![codecov](https://codecov.io/gh/dssg/architect/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/architect)
[![codeclimate](https://codeclimate.com/github/dssg/architect.png)](https://codeclimate.com/github/dssg/architect)

In order to run classification algorithms on source data, this data must be properly organized into design matrices. Converting cleaned data into these matrices is not a trivial task; the process of creating the needed features and labels for an experiment from source data can be complicated, creating the matrices themselves out of features and labels can be inefficient, and there is opportunity at each step to leak data backwards in time to give model trained on a matrix an unfair advantage.

The Architect addresses these issues with functionality aimed at all tasks between cleaned source data (in a PostgreSQL database) and design matrices.

## Components

- [LabelGenerator](architect/label_generators.py): Create binary labels suitable for a design matrix by querying a database table containing outcome events.
- [FeatureGenerator](architect/feature_generators.py): Create aggregate features suitable for a design matrix from a set of database tables containing events. Uses [collate](https://github.com/dssg/collate/) to build aggregation SQL queries.
- [FeatureGroupCreator](architect/feature_group_creator.py), [FeatureGroupMixer](architect/feature_group_mixer.py): Create groupings of features, and mix them using different strategies (like 'leave one out') to test their effectiveness.
- [Planner](architect/planner.py), [Builder](architect/builders.py): Build all design matrices needed for an experiment, taking into account different labels, state configurations, and feature groups.

In addition to being usable individually to assist in different aspects of building matrices in your project, the Architect components are integrated in [triage](https://github.com/dssg/triage) as a part of an entire modeling experiment that incorporates later tasks like model training and testing.

## Distributing, Building &amp; Testing

The Architect is a Python package distributable via `setuptools`. It may be installed directly using `easy_install` or `pip`, or listed as a dependency of another package (namely `triage`), under the package name `matrix-architect`.

To build this package for development, its dependencies may be installed using `pip`:

    pip install -r requirements_dev.txt

(or, without test and development dependencies, using **requirements.txt**).

And, having built for development, to run tests:

    pytest
