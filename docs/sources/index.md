# Triage

Risk modeling and prediction

[![Build Status](https://travis-ci.org/dssg/triage.svg?branch=master)](https://travis-ci.org/dssg/triage)
[![codecov](https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/triage)
[![codeclimate](https://codeclimate.com/github/dssg/triage.png)](https://codeclimate.com/github/dssg/triage)


Predictive analytics projects require the coordination of many different tasks, such as feature generation, classifier training, evaluation, and list generation. These tasks are complicated in their own right, but in addition have to be combined in different ways throughout the course of the project. 

Triage aims to provide interfaces to these different phases of a project, such as an `Experiment`. Each phase is defined by configuration specific to the needs of the project, and an arrangement of core data science components that work together to produce the output of that phase.

The first phase implemented in triage is the `Experiment`. An experiment represents the initial research work of creating design matrices from source data, and training/testing/evaluating a model grid on those matrices. At the end of the experiment, a relational database with results metadata is populated, allowing for evaluation by the researcher.


## Running an Experiment
See the [Running an Experiment](experiments/running.md) documentation.

## Upgrading an Experiment config
[v3/v4 -> v5](experiments/upgrade-to-v5.md)


## Background

Triage is developed at the University of Chicago's [Center For Data Science and Public Policy](http://dsapp.uchicago.edu). We created it in response to commonly occuring challenges we've encountered and patterns we've developed while working on projects for our partners.

## Major Components Used by Triage

Triage makes use of many core data science components developed at DSaPP. These components can be useful in their own right, and are worth checking out if 

* [Architect](https://github.com/dssg/architect): Plan, design and build train and test matrices. Includes feature and label generation.
* [Collate](https://github.com/dssg/collate): Aggregation SQL Query Builder. This is used by the Architect to build features.
* [Timechop](https://github.com/dssg/timechop): Generate temporal cross-validation time windows for matrix creation
* [Metta-Data](https://github.com/dssg/metta-data): Train and test matrix storage
* [Catwalk](https://github.com/dssg/catwalk): Training, testing, and evaluating machine learning classifier models
* [Results Schema](https://github.com/dssg/results-schema): Generate a database schema suitable for storing the results of modeling runs


## Design Goals

There are two overarching design goals for Triage:

- All configuration necessary to run the full experiment from the external interface (ie, Experiment subclasses) from beginning to end must be easily serializable and machine-constructable, to allow the eventual development of tools for users to design experiments. 

- All core functionality must be usable outside of a specific pipeline context or workflow manager. There are many good workflow managers; everybody has their favorite, and core functionality should not be designed to work with specific execution expectations.


## Future Plans

- Generation and Management of lists (ie for inspections) by various criteria
- Integration of components with various workflow managers, like [Drain](https://github.com/dssg/drain) and [Luigi](https://github.com/spotify/luigi).
- Comprehensive leakage testing of an experiment's modeling run
- Feature Generation Wizard
