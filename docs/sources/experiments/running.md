# Running an Experiment

## Prerequisites

To use a Triage experiment, you first need:

- Python 3.5
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk (or S3) to store the needed matrices and models for your experiment
- An experiment definition (see [Defining an Experiment](defining.md))


## Instantiating an Experiment

An `Experiment` class, once instantiated, provides access to a variety of useful pieces of information about the experiment, as well as the ability to run it and get results.

First, we'll look at how to instantiate the Experiment.

```
    SingleThreadedExperiment(
        config=experiment_config,
        db_engine=triage.create_engine(...),
        model_storage_class=FSModelStorageEngine,
        project_path='/path/to/directory/to/save/data'
    )
```

These lines are a bit dense: what is happening here?

- `SingleThreadedExperiment`:  There are different Experiment classes available in ``triage.experiments`` to use, and they each represent a different way of executing the experiment, which we'll talk about in more detail later. The simplest (but slowest) is the `SingleThreadedExperiment`.
- `config=experiment_config`: The bulk of the work needed in designing an experiment will be in creating this experiment configuration. See [defining.md](Defining an Experiment); more detailed instructions on each section are located in the example file. Generally these would be easiest to store as a file (or multiple files that you construct together) like that YAML file, but the configuration is passed in dict format to the Experiment constructor and you can store it however you wish.
- `db_engine=triage.create_engine(...)`: A SQLAlchemy database engine. This will be used both for querying your source tables and writing results metadata. To create this, we recommend using `triage.create_engine`. It will create a serializable version of the engine that will be fully reconstructed in multiprocess contexts. If you pass a regular SQLAlchemy engine, in these contexts the engine will be reconstructed with the *URL only*, which may cancel other settings you have used to configure your engine.
- `model_storage_class=FSModelStorageEngine`: The path to a model storage engine class. A model storage engine class is one that wraps all model storage I/O operations, allowing storage medium (e.g. filesystem, S3) to be switched easily. Triage provides a few of these in [triage.component.catwalk.storage](../src/triage/component/catwalk/storage.py). Shown here is the `FSModelStorageEngine`, which is a good place to start.
- `project_path='/path/to/directory/to/save/data'`: The path to where you would like to store design matrices and trained models. May be an s3 path (e.g. s3://bucket-name/project-directory), in which case s3 will be used to store matrices and models.

With that in mind, a more full version of the experiment instantiation might look like this

```
    import yaml
    import logging

    from triage import create_engine
    from triage.component.catwalk.storage import FSModelStorageEngine
    from triage.experiments import SingleThreadedExperiment

    with open('my_experiment_config.yaml') as f:
    	experiment_config = yaml.load(f)
    with open('my_database_creds') as f:
    	db_connection_string = yaml.load(f)['db_connection_string']

    logging.basicConfig(level=logging.INFO)

    experiment = SingleThreadedExperiment(
        config=experiment_config,
        db_engine=create_engine(db_connection_string),
        model_storage_class=FSModelStorageEngine,
        project_path='/home/research/myproject'
    )
```

## Validating an Experiment

Configuring an experiment is very complicated, and running an experiment can take a long time as data scales up. If there are any misconfigured values, it's going to help out a lot to figure out what they are before we run the Experiment. So we recommend running the `.validate()` method on the Experiment first. If any problems are detectable in your Experiment, either in configuration or the database tables referenced by it, this method will throw an exception. For instance, if I refer to the 'cat_complaints' table in a feature aggregation but it doesn't exist, I'll see something like this:

```
    experiment.validate()

    (Pdb) experiment.validate()
    *** ValueError: from_obj query does not run.
    from_obj: "cat_complaints"
    Full error: (psycopg2.ProgrammingError) relation "cat_complaints" does not exist
    LINE 1: explain select * from cat_complaints
                                  ^
     [SQL: 'explain select * from cat_complaints']
```

If the validation runs without any errors, you should see a success message (either in your log or console). At this point, the Experiment should be ready to run.

We'd like to add more validations for common misconfiguration problems over time. If you got an unexpected error that turned out to be related to a confusing configuration value, help us out by adding to the [validation module](triage/experiments/validate.py) and submitting a pull request!


## Running an Experiment

Once you're at this point, running the experiment is simple:

```
    experiment.run()
```

This will run the entire experiment. This could take a while, so we recommend checking logging messages (INFO level will catch a lot of useful information) and keeping an eye on its progress.

## Running parts of an Experiment

If you would like to run parts of the Experiment one by one and look at their outputs, you can do so. To reproduce the entire Experiment piece by piece, you can run the following:

- `experiment.split_definitions` will parse temporal config and create time splits.

- `experiment.generate_sparse_states()` will use the cohort config and as of dates from the temporal config to generate an internal table keeping track of what entities are in the cohort on different dates.

- `experiment.generate_labels()` will use the label config and as of dates from the temporal config to generate an internal labels table.

- `experiment.generate_preimputation_features()` will use the feature aggregation config and as of dates from the temporal config to generate internal features tables.

- `experiment.generate_imputed_features()` will use the imputation sections of the feature aggregation config and the results from the preimputed features to create internal imputed features tables

- `experiment.build_matrices()` will use all of the internal tables generated before this point, along with feature grouping config, to generate all needed matrices.

- `experiment.train_and_test_models()` will use the generated matrices, grid config and evaluation metric config to train and test all needed models.


## Evaluating results of an Experiment

After the experiment run, a variety of schemas and tables will be created and populated in the configured database:

* model_metadata.experiments - The experiment configuration and a hash
* model_metadata.models - A model describes a trained classifier; you'll have one row for each trained file that gets saved.
* model_metadata.model_groups - A model groups refers to all models that share parameters like classifier type, hyperparameters, etc, but *have different training windows*. Look at these to see how classifiers perform over different training windows.
* model_metadata.matrices - Each matrix that was used for training and testing has metadata written about it such as the matrix hash, length, and time configuration.
* train_results.feature_importances - The sklearn feature importances results for each trained model
* train_results.predictions - Prediction probabilities for train matrix entities generated against trained models
* train_results.evaluations - Metric scores of trained models on the training data.
* test_results.predictions - Prediction probabilities for test matrix entities generated against trained models
* test_results.evaluations - Metric scores of trained models over given testing windows
* test_results.individual_importances - Individual feature importance scores for test matrix entities.

Here's an example query, which returns the top 10 model groups by precision at the top 100 entities:

```
    select
    	model_groups.model_group_id,
    	model_groups.model_type,
    	model_groups.hyperparameters,
    	max(test_evaluations.value) as max_precision
    from model_metadata.model_groups
    	join model_metadata.models using (model_group_id)
    	join test_results.test_evaluations using (model_id)
    where
    	metric = 'precision@'
    	and parameter = '100_abs'
    group by 1,2,3
    order by 4 desc
    limit 10
```


## Restarting an Experiment

If an experiment fails for any reason, you can restart it. Each matrix and each model file is saved with a filename matching a hash of its unique attributes, so when the experiment is rerun, it will by default reuse the matrix or model instead of rebuilding it. If you would like to change this behavior and replace existing versions of matrices and models, set `replace=True` in the Experiment constructor.

## Inspecting an Experiment before running

Before you run an experiment, you can inspect properties of the Experiment object to ensure that it is configured in the way you want. Some examples:

- `experiment.all_as_of_times` for debugging temporal config. This will show all dates that features and labels will be calculated at.
- `experiment.feature_dicts` will output a list of feature dictionaries, representing the feature tables and columns configured in this experiment
- `experiment.matrix_build_tasks` will output a list representing each matrix that will be built.

## Experiment Classes

- *SingleThreadedExperiment*: An experiment that performs all tasks serially in a single thread. Good for simple use on small datasets, or for understanding the general flow of data through a pipeline.
- *MultiCoreExperiment*: An experiment that makes use of the multiprocessing library to parallelize various time-consuming steps. Takes an `n_processes` keyword argument to control how many workers to use.
- *RQExperiment*: An experiment that makes use of the python-rq library to enqueue individual tasks onto the default queue, and wait for the jobs to be finished before moving on. python-rq requires Redis and any number of worker processes running the Triage codebase. Triage does not set up any of this needed infrastructure for you. Available through the RQ extra ( `pip install triage[rq]` )
