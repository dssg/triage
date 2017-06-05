# Triage

Risk modeling and prediction

[![Build Status](https://travis-ci.org/dssg/triage.svg?branch=master)](https://travis-ci.org/dssg/triage)
[![codecov](https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/triage)
[![codeclimate](https://codeclimate.com/github/dssg/triage.png)](https://codeclimate.com/github/dssg/triage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Basic Usage

The simplest usage of triage is to use a pipeline. Triage pipelines are aimed at handling label generation, generating features (using [collate](https://github.com/dssg/collate)), creating training/test splits (using [timechop](https://github.com/dssg/timechop)), training models, testing models, and calculating metrics. Triage pipelines are generally run after data is loaded into a Postgres database and cleaned. The model metadata, feature importances, predictions, and model metrics are saved to the `results` schema in the given database.  Different pipelines are available, which control execution in different ways.  The SerialPipeline, for instance, is the simplest but slowest, while the LocalParallelPipeline makes use of multiple cores on the local machine.


### Construct a Pipeline

To construct one of these pipeline objects, you need the following arguments:

- An experiment configuration. This contains time-splitting, feature-generation, grid-search, and model-scoring configurations. An up-to-date example is at [example_experiment_config.yaml](https://github.com/dssg/triage/blob/master/example_experiment_config.yaml); more detailed instructions are located in the example file. This is passed in dict format to the pipeline constructor.
- A SQLAlchemy Postgres database engine. There is a convenience wrapper at triage.db.connect() that reads from a local database.yaml file, but you can supply an engine created however you like.
- A model-storage engine class from the `triage.storage` module. The model-storage engines are wrappers around various ways to cache models, and are exclusively used for model caching inside triage components. Current implementations include InMemoryModelStorageEngine (no caching), FSModelStorageEngine (on local or mounted disk), and S3ModelStorageEngine (direct to AWS s3). FSModelStorageEngine is recommended to start.
- Pick a project path. This is used by the persistent-storage engines.


### Run a Pipeline

The pipeline has one main method: `.run()`. This will perform all work, from label generation through model metric creation.

```
    from triage.db import connect
    from triage.storage import FSModelStorageEngine
    from triage.pipelines import SerialPipeline

	with open('example_experiment_config.yaml') as f:
		experiment_config = yaml.load(f)
	pipeline = SerialPipeline(
		config=experiment_config,
		db_engine=connect(),
		model_storage_class=FSModelStorageEngine,
		project_path='/path/to/cached_models/test_project'
	)

	pipeline.run()
```


### Evaluating results of a pipeline

After the experiment run, a results schema will be created in the configured database with the following tables:
- experiments - The experiment configuration and a hash
- model_groups - Definitions of 'model groups'; model groups are defined by all models that share all parameters except for training date
- models - Each model, including training date
- feature_importances - The sklearn feature importances results for each model
- predictions - Prediction probabilities for entities generated against trained models
- evaluations - Metric scores of trained models over given testing windows


## Advanced Use

The current method of changing behavior beyond what can be controlled in the experiment config file is to use the inner components (e.g. FeatureGenerator, ModelTrainer, etc). Each component is instantiated using a certain section of the same configuration that is passed to a pipeline. Support will be added in the future passing in subclassed components like an alternate label generator, and usage in various workflow managers. It can also be useful to copy a pipeline file into your project and tweak to your liking. If you do this to implement a new feature, please submit a [feature request](https://github.com/dssg/triage/blob/60ecb0cc3ab7b1c0aa99917c624f794d20fc9f15/CONTRIBUTING.rst) so others can use it!

## Components

- *Time Choppers* (provided by [Timechop](https://github.com/dssg/timechop)): split user-friendly time configuration into full temporal configuration for matrices.
- *Label Generators*: define a labels table based on events. For instance, BinaryLabelGenerator defines binary labels based on an events table specified in configuration.
- *Feature Generators*: generate feature tables based on source tables and configuration. Uses [collate](https://github.com/dssg/collate) to generate aggregate features.
- *Feature Dictionary Creators*: based on a list of feature tables, store all feature names into a serializable dictionary based on feature table.
- *Feature Group Creators and Feature Group Mixers*: define groups of feature tables based on configurable criteria (such as source table), and create many subsets of the feature dictionary to test hypotheses (ie, does leaving out certain features affect results?)
- *Architect* (provided by [Timechop](https://github.com/dssg/timechop)): combine computed features and labels into design matrices. Uses [metta-data](https://github.com/dssg/metta-data) for storing matrices with useful metadata, using the hash of the matrix metadata to avoid an explosion of storage space.
- *Model Trainers*: train a configured experiment grid on pre-made design matrices, and store each model's metadata and feature importances in a database.
- *Predictors*: given a trained model and another matrix (ie, a test matrix), generate prediction probabilities and store them in a database.
- *Model Scorers*: given a set of model prediction probabilities, generate metrics (for instance, precision and recall at various thresholds) and store them in a database.


## Pipelines

- *SerialPipeline*: a single-threaded pipeline. Good for simple use on small datasets, or for understanding the general flow of data through a pipeline.
- *LocalParallelPipeline*: A pipeline that makes use of the multiprocessing library to parallelize various time-consuming steps. Takes an n_processes keyword argument to control how many workers to use.


## Design Goals

There are two overarching design goals for Triage:

- All configuration necessary to run the full pipeline from the external interface (ie, Pipeline subclasses) from beginning to end must be serializable. Most of this is common in data science pipelines, but it also means that feature definition must not live in code.
- All core functionality must be usable outside of a specific pipeline context or workflow manager. There are many good workflow managers, everybody has their favorite, and core functionality should not be designed to work with specific execution expectations.


## Future Plans

- Generation and Management of lists (ie for inspections) by various criteria
- Integration of components with various workflow managers, like [Drain](https://github.com/dssg/drain) and [Luigi](https://github.com/spotify/luigi).
- Comprehensive leakage testing of a pipeline's experiment run
- Feature Generation Wizard
