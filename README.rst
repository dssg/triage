======
Triage
======

Risk modeling and prediction

.. image:: https://travis-ci.org/dssg/triage.svg?branch=master
   :target: https://travis-ci.org/dssg/triage

.. image:: https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/dssg/triage

.. image:: https://codeclimate.com/github/dssg/triage.png
   :target: https://codeclimate.com/github/dssg/triage

Building systems that use predictive models require answering many design decisions, turning them into modeling choices, and technical tasks. Questions such as cohort selection, unit of analysis determination, outcome determination, feature (explanantory variables) generation, model/classifier training, evaluation, selection, and list generation are often complicated and hard to choose apriori. In addition, once these choices are made, they have to be combined in different ways throughout the course of a project.

Triage aims to help solve these problems by:

1. Guiding users (data scientists, analysts, researchers) through these design choices by highlighting operational use questions that are important.

2. Providing interfaces to these different phases of a project, such as an ``Experiment``. Each phase is defined by a configuration (corresponding to a design choice) specific to the needs of the project, and an arrangement of core data science components that work together to produce the output of that phase. 



Installation
============

Prerequisites
-------------

To use Triage, you first need:

- Python 3+
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk, (or for example in Amazon Web Services's S3), to store the needed matrices and models for your experiment

Building
--------

Triage is a Python package distributable via ``setuptools``. It may be installed directly using ``easy_install`` or ``pip``, or named as a dependency of another package as ``triage``.

To build this package (without installation), its dependencies may alternatively be installed from the terminal using ``pip``::

    pip install -r requirement/main.txt

Testing
-------

To add test (and development) dependencies, use **test.txt**::

    pip install -r requirement/test.txt [-r requirement/dev.txt]

Then, to run tests::

    pytest

Development
-----------

To quickly bootstrap a development environment, having cloned the repository, invoke the executable ``develop`` script from your system shell::

    ./develop

A "wizard" will suggest set-up steps and optionally execute these, for example::

    (install) begin

    (pyenv) installed ✓

    (python-3.6.2) installed ✓

    (virtualenv) installed ✓

    (activation) installed ✓

    (libs) install?
    1) yes, install {pip install -r requirement/main.txt -r requirement/test.txt -r requirement/dev.txt}
    2) no, ignore
    #? 1

Experiment
==========

The first phase implemented in Triage is the ``Experiment``. An experiment represents the initial research work of creating design matrices from source data, and training/testing/evaluating a model grid on those matrices. At the end of the experiment, a relational database with results metadata is populated, allowing for evaluation by the researcher.


Documentation
---------------------------
- `Dirty Duck Tutorial <https://dssg.github.io/dirtyduck/>`_
- `Running an Experiment <https://dssg.github.io/triage/experiments/running>`_
- `Experiment Algorithm Deep Dive <https://dssg.github.io/triage/experiments/algorithm>`_
- `Experiment Config v5 Upgrade Guide <https://dssg.github.io/triage/experiments/upgrade-to-v5>`_


Background
==========

Triage is developed at the University of Chicago's `Center For Data Science and Public Policy <http://dsapp.uchicago.edu>`_. We created it in response to commonly occuring challenges we've encountered and patterns we've developed while working on projects for our partners.

Major Components Used by Triage
===============================

Triage makes use of many core data science components developed at DSaPP. These components can be useful in their own right, and are worth checking out if you'd like to make use of a subset of Triage's functionality in an existing pipeline.

Components Within Triage
------------------------

* `Architect <src/triage/component/architect>`_: Plan, design and build train and test matrices. Includes feature and label generation.
* `Catwalk <src/triage/component/catwalk>`_: Training, testing, and evaluating machine learning classifier models
* `Collate <src/triage/component/collate>`_: Aggregation SQL Query Builder. This is used by the Architect to build features.
* `Timechop <src/triage/component/timechop>`_: Generate temporal cross-validation time windows for matrix creation
* `Results Schema <src/triage/component/results_schema>`_: Generate a database schema suitable for storing the results of modeling runs

Design Goals
============

There are two overarching design goals for Triage:

- All configuration necessary to run the full experiment from the external interface (ie, Experiment subclasses) from beginning to end must be easily serializable and machine-constructable, to allow the eventual development of tools for users to design experiments.

- All core functionality must be usable outside of a specific pipeline context or workflow manager. There are many good workflow managers; everybody has their favorite, and core functionality should not be designed to work with specific execution expectations.

Future Plans
============

- Generation and Management of lists (ie for inspections) by various criteria
- Integration of components with various workflow managers, like `Drain <https://github.com/dssg/drain>`_ and `Luigi <https://github.com/spotify/luigi>`_.
- Comprehensive leakage testing of an experiment's modeling run
- Feature Generation Wizard
