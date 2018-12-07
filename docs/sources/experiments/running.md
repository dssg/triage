# Running an Experiment

## Prerequisites

To use a Triage experiment, you first need:

- Python 3.5
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk (or S3) to store the needed matrices and models for your experiment
- An experiment definition (see [Defining an Experiment](defining.md))



You may run a Triage experiment two ways: through the Triage command line tool, or through instantiating an Experiment object in your own Python code and running it. The rest of this document will introduce experiment runs of increasing complexity, through both the CLI and Python interfaces.


## Simple Example

To run an experiment, you need to tell triage at a minimum where to find the experiment file (in YAML format), and how to connect to the database, In this simple example, we're assuming that the experiment will be run with only one process, and that the matrices and models should be stored on the local filesystem.

### CLI

The Triage CLI defaults database connection information to a file stored in 'database.yaml', so with this you can omit any mention of the database. In addition, if you leave out the project path. In addition, the 'project path' (where matrices and models are stored) defaults to the current working directory. So this is the simplest possible invocation:

```bash
triage experiment example/config/experiment.yaml
```

If you have the database information stored somewhere else, you may pass it to the top-level 'triage' command:

```bash
triage -d mydbconfig.yaml experiment example/config/experiment.yaml
```

Assuming you want the matrices and models stored somewhere else, pass it as the `--project-path`:

```bash
triage -d mydbconfig.yaml experiment example/config/experiment.yaml --project-path '/path/to/directory/to/save/data'
```

### Python

When running an experiment in Python, the database information is passed in the form of a SQLAlchemy database engine, and the experiment information is passed as a dictionary rather as YAML specifically.

```python
from triage.experiments import SingleThreadedExperiment

experiment = SingleThreadedExperiment(
    config=experiment_config, # a dictionary
    db_engine=create_engine(...), # http://docs.sqlalchemy.org/en/latest/core/engines.html
    project_path='/path/to/directory/to/save/data'
)
experiment.run()
```

## Multicore example

Triage also offers the ability to locally parallelize both CPU-heavy and database-heavy tasks. Triage uses the [pebble](https://pythonhosted.org/Pebble) library to perform both of these, but they are separately configurable as the database tasks will more likely be bounded by the number of connections/cores available on the database server instead of the number of cores available on the experiment running machine.

### CLI

The Triage CLI allows parallelization to be specified through the `--n-processes` and `--n-db-processes` parameters.

```bash
triage experiment example/config/experiment.yaml --project-path '/path/to/directory/to/save/data' --n-db-processes 4 --n-processes 8
```

### Python

In Python, you can use the `MultiCoreExperiment` instead of the `SingleThreadedExperiment`, and similarly pass the `n_processes` and `n_db_processes` parameters. We also recommend using `triage.create_engine`. It will create a serializable version of the engine that will be fully reconstructed in multiprocess contexts. If you pass a regular SQLAlchemy engine, in these contexts the engine will be reconstructed with the [database URL only](http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls), which may cancel other settings you have used to configure your engine.

```python

from triage.experiments import MultiCoreExperiment
from triage import create_engine

experiment = MultiCoreExperiment(
    config=experiment_config, # a dictionary
    db_engine=create_engine(...),
    project_path='/path/to/directory/to/save/data',
    n_db_processes=4,
    n_processes=8,
)
experiment.run()

```

The [pebble](https://pythonhosted.org/Pebble) library offers an interface around Python3's `concurrent.futures` module that adds in a very helpful tool: watching for killed subprocesses . Model training (and sometimes, matrix building) can be a memory-hungry task, and Triage can not guarantee that the operating system you're running on won't kill the worker processes in a way that prevents them from reporting back to the parent Experiment process. With Pebble, this occurrence is caught like a regular Exception, which allows the Process pool to recover and include the information in the Experiment's log.

## Using S3 to store matrices and models

Triage can operate on different storage engines for matrices and models, and besides the standard filesystem engine comes with S3 support out of the box. To use this, just use the `s3://` scheme for your `project_path` (this is similar for both Python and the CLI).


### CLI

```bash
triage experiment example/config/experiment.yaml --project-path 's3://bucket/directory/to/save/data'
```

### Python

```python
from triage.experiments import SingleThreadedExperiment

experiment = SingleThreadedExperiment(
    config=experiment_config, # a dictionary
    db_engine=create_engine(...),
    project_path='s3://bucket/directory/to/save/data'
)
experiment.run()

```

## Using HDF5 as a matrix storage format

Triage by default uses CSV format to store matrices, but this can take up a lot of space. However, this is configurable.  Triage ships with an HDF5 storage module that you can use.

### CLI

On the command-line, this is configurable using the `--matrix-format` option, and supports `csv` and `hdf`.

```bash
triage experiment example/config/experiment.yaml --matrix-format hdf
```

### Python

In Python, this is configurable using the `matrix_storage_class` keyword argument. To allow users to write their own storage modules, this is passed in the form of a class. The shipped modules are in `triage.component.catwalk.storage`. If you'd like to write your own storage module, you can use the [existing modules](https://github.com/dssg/triage/blob/master/src/triage/component/catwalk/storage.py) as a guide.

```python
from triage.experiments import SingleThreadedExperiment
from triage.component.catwalk.storage import HDFMatrixStore

experiment = SingleThreadedExperiment(
    config=experiment_config
    db_engine=create_engine(...),
    matrix_storage_class=HDFMatrixStore,
    project_path='/path/to/directory/to/save/data',
)
experiment.run()
```

Note: The HDF storage option is *not* compatible with S3.

## Validating an Experiment

Configuring an experiment is complex, and running an experiment can take a long time as data scales up. If there are any misconfigured values, it's going to help out a lot to figure out what they are before we run the Experiment. So when you have completed your experiment config and want to test it out, it's best to validate the Experiment first. If any problems are detectable in your Experiment, either in configuration or the database tables referenced by it, this method will throw an exception. For instance, if I refer to the `cat_complaints` table in a feature aggregation but it doesn't exist, I'll see something like this:

```
*** ValueError: from_obj query does not run.
from_obj: "cat_complaints"
Full error: (psycopg2.ProgrammingError) relation "cat_complaints" does not exist
LINE 1: explain select * from cat_complaints
                              ^
 [SQL: 'explain select * from cat_complaints']
```


### CLI

The CLI, by default, validates before running. You can tweak this behavior, and make it not validate, or make it *only* validate.

```bash
triage experiment example/config/experiment.yaml --project-path '/path/to/directory/to/save/data' --no-validate
```

```bash
triage experiment example/config/experiment.yaml --project-path '/path/to/directory/to/save/data' --validate-only
```

#### Python

Experiments expose a `validate` method that can be run as needed. Experiment instantiation doesn't change from the run examples at all.

```python
experiment.validate()
```

By default, the `validate` method will stop as soon as it encounters an error ('strict' mode). If you would like it to validate each section without stopping (i.e. if you have only written part of the experiment configuration), call `validate(strict=False)` and all of the errors will be changed to warnings.

We'd like to add more validations for common misconfiguration problems over time. If you got an unexpected error that turned out to be related to a confusing configuration value, help us out by adding to the [validation module](https://github.com/dssg/triage/blob/master/src/triage/experiments/validate.py) and submitting a pull request!


## Restarting an Experiment

If an experiment fails for any reason, you can restart it.

By default, all work will be recreated. This includes label queries, feature queries, matrix building, model training, etc. However, if you pass the `replace=False` keyword argument, the Experiment will reuse what work it can.

- Labels Table: The Experiment keeps a labels table namespaced by its experiment hash, and within that will check on a per-`as_of_date`/`label timespan` level whether or not there are *any* existing rows, and skip the label query if so. For this reason, it is *not* aware of specific entities or source events so if the label query has changed or the source data has changed, you will not want to set `replace` to False. Don't expect too much reuse from this, however, as the table is experiment-namespaced. Essentially, this will only reuse data if the same experiment was run prior and failed part of the way through label generation. 
- Features Tables: The Experiment will check on a per-table basis whether or not it exists, and skip the feature generation if so. Each 'table' maps to a feature aggregation in your experiment config, so if you have modified any source data that affects that feature aggregation, added any features to that aggregation, or changed any `temporal_config` so there are more `as_of_dates`, you won't want to set `replace` to False.
- Matrix Building: Each matrix's metadata is hashed to create a unique id. If a file exists in storage with that hash, it will be reused.
- Model Training: Each model's metadata (which includes its train matrix's hash) is hashed to create a unique id. If a file exists in storage with that hash, it will be reused.


### CLI

```bash
triage experiment example/config/experiment.yaml --project-path '/path/to/directory/to/save/data' --replace
```

### Python

```python
from triage.experiments import SingleThreadedExperiment

experiment = SingleThreadedExperiment(
    config=experiment_config, # a dictionary
    db_engine=create_engine(...),
    project_path='s3://bucket/directory/to/save/data',
    replace=True
)
experiment.run()
```

## Running parts of an Experiment

If you would like incrementally build, or just incrementally run parts of the Experiment look at their outputs, you can do so. Running a full experiment requires the [experiment config](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) to be filled out, but when you're getting started using Triage it can be easier to build the experiment piece by piece and see the results as they come in. Make sure logging is set to INFO level before running this to ensure you get all the log messages.

Running parts of an experiment is only supported through the Python interface.


### Python

1. `experiment.run()` will run until it no longer has enough configuration to proceed. You will see information in the logs telling you about the steps it was able to perform. If you initialize the Experiment with `cleanup=False`, you can view the intermediate tables that are built. They are modified with the experiment hash that the experiment calculates, but this will be printed out in the log messages.

	- `labels_*<experiment_hash>*` for the labels generated per entity and as of date.
	- `tmp_sparse_states_*<experiment_hash>*` for the membership in each cohort per entity and as_of_date

2. To reproduce the entire Experiment piece by piece, you can run the following. Each one of these methods requires some portion of [experiment config](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) to be passed:

	- `experiment.split_definitions` will parse temporal config and create time splits. It only requires `temporal_config`.

	- `experiment.generate_cohort()` will use the cohort config and as of dates from the temporal config to generate an internal table keeping track of what entities are in the cohort on different dates. It requires `temporal_config` and `cohort_config`.

	- `experiment.generate_labels()` will use the label config and as of dates from the temporal config to generate an internal labels table. It requires `temporal_config` and `label_config`.

	- `experiment.generate_preimputation_features()` will use the feature aggregation config and as of dates from the temporal config to generate internal features tables. It requires `temporal_config` and `feature_aggregations`.

	- `experiment.generate_imputed_features()` will use the imputation sections of the feature aggregation config and the results from the preimputed features to create internal imputed features tables. It requires `temporal_config` and `feature_aggregations`.

	- `experiment.build_matrices()` will use all of the internal tables generated before this point, along with feature grouping config, to generate all needed matrices.  It requires `temporal_config`, `cohort_config`, `label_config`, and `feature_aggregations`, though it will also use `feature_group_definitions`, `feature_group_strategies`, and `user_metadata` if present.

	- `experiment.train_and_test_models()` will use the generated matrices, grid config and evaluation metric config to train and test all needed models. It requires all configuration keys.


## Evaluating results of an Experiment

After the experiment run, a variety of schemas and tables will be created and populated in the configured database:

* model_metadata.experiments - The experiment configuration and a hash
* model_metadata.matrices - Each train or test matrix that is built has a row here, with some basic metadata
* model_metadata.experiment_matrices - A many-to-many table between experiments and matrices. This will have a row if the experiment used the matrix, regardless of whether or not it had to build it
* model_metadata.models - A model describes a trained classifier; you'll have one row for each trained file that gets saved.
* model_metadata.experiment_models - A many-to-many table between experiments and models. This will have a row if the experiment used the model, regardless of whether or not it had to build it
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
    	join test_results.evaluations using (model_id)
    where
    	metric = 'precision@'
    	and parameter = '100_abs'
    group by 1,2,3
    order by 4 desc
    limit 10
```

## Inspecting an Experiment before running

Before you run an experiment, you can inspect properties of the Experiment object to ensure that it is configured in the way you want. Some examples:

- `experiment.all_as_of_times` for debugging temporal config. This will show all dates that features and labels will be calculated at.
- `experiment.feature_dicts` will output a list of feature dictionaries, representing the feature tables and columns configured in this experiment
- `experiment.matrix_build_tasks` will output a list representing each matrix that will be built.

## Experiment Classes

- *SingleThreadedExperiment*: An experiment that performs all tasks serially in a single thread. Good for simple use on small datasets, or for understanding the general flow of data through a pipeline.
- *MultiCoreExperiment*: An experiment that makes use of the pebble library to parallelize various time-consuming steps. Takes an `n_processes` keyword argument to control how many workers to use.
- *RQExperiment*: An experiment that makes use of the python-rq library to enqueue individual tasks onto the default queue, and wait for the jobs to be finished before moving on. python-rq requires Redis and any number of worker processes running the Triage codebase. Triage does not set up any of this needed infrastructure for you. Available through the RQ extra ( `pip install triage[rq]` )
