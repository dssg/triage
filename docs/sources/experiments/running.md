# Running an Experiment

## Prerequisites

To use a Triage experiment, you first need:

- Python 3.5
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk (or S3) to store the needed matrices and models for your experiment
- An experiment definition (see [Experiment configuration](experiment-config.md))



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

Either way you run it, you are likely to see a bunch of log output.  Once the feature/cohor/label/matrix building is done and the experiment has moved onto modeling, check out the `triage_metadata.models` and `test_results.evaluations` tables as data starts to come in. You'll see the simple models (Decision Trees, Scaled Logistic Regression, baselines) populate first, followed by your big models, followed by the rest. You can start to look at the simple model results first to get a handle on what basic classifiers can do for your feature space while you wait for the Random Forests to run.

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

The python interface will also validate by default when running an experiment. If you would prefer to skip this step, you can pass `skip_validation=True` when constructing your experiment.

You can also run this validation step directly. Experiments expose a `validate` method that can be run as needed. Experiment instantiation doesn't change from the run examples at all.

```python
experiment.validate()
```

By default, the `validate` method will stop as soon as it encounters an error ('strict' mode). If you would like it to validate each section without stopping (i.e. if you have only written part of the experiment configuration), call `validate(strict=False)` and all of the errors will be changed to warnings.

We'd like to add more validations for common misconfiguration problems over time. If you got an unexpected error that turned out to be related to a confusing configuration value, help us out by adding to the [validation module](https://github.com/dssg/triage/blob/master/src/triage/experiments/validate.py) and submitting a pull request!


## Restarting an Experiment

If an experiment fails for any reason, you can restart it.

By default, all work will be recreated. This includes label queries, feature queries, matrix building, model training, etc. However, if you pass the `replace=False` keyword argument, the Experiment will reuse what work it can.

- Cohort Table: The Experiment refers to a cohort table namespaced by the cohort name and a hash of the cohort query, and in that way allows you to reuse cohorts between different experiments if their label names and queries are identical. When referring to this table, it will check on an as-of-date level whether or not there are any existing rows for that date, and skip the cohort query for that date if so. For this reason, it is *not* aware of specific entities or source events so if the source data has changed, ensure that `replace` is set to True. 
- Labels Table: The Experiment refers to a labels table namespaced by the label name and a hash of the label query, and in that way allows you to reuse labels between different experiments if their label names and queries are identical. When referring to this table, it will check on a per-`as_of_date`/`label timespan` level whether or not there are *any* existing rows, and skip the label query if so. For this reason, it is *not* aware of specific entities or source events so if the label query has changed or the source data has changed, ensure that `replace` is set to True.
- Features Tables: The Experiment will check on a per-table basis whether or not it exists and contains rows for the entire cohort, and skip the feature generation if so. It does not look at the column list for the feature table or inspect the feature data itself. So, if you have modified any source data that affects a feature aggregation, or added any columns to that aggregation, you won't want to set `replace` to False. However, it is cohort-and-date aware so you can change around your cohort and temporal configuration safely.
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

## Optimizing an Experiment

### Skipping Prediction Syncing
By default, the Experiment will save predictions to the database. This can take a long time if your test matrices have a lot of rows, and isn't quite necessary if you just want to see the high-level performance of your grid. By switching `save_predictions` to `False`, you can skip the prediction saving. You'll still get your evaluation metrics, so you can look at performance. Don't worry, you can still get your predictions back later by rerunning the Experiment later at default settings, which will find your already-trained models, generate predictions, and save them.

CLI: `triage experiment myexperiment.yaml --no-save-predictions`

Python: `SingleThreadedExperiment(..., save_predictions=False)`

## Running parts of an Experiment

If you would like incrementally build, or just incrementally run parts of the Experiment look at their outputs, you can do so. Running a full experiment requires the [experiment config](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) to be filled out, but when you're getting started using Triage it can be easier to build the experiment piece by piece and see the results as they come in. Make sure logging is set to INFO level before running this to ensure you get all the log messages. Additionally, because the default behavior of triage is to run config file validation (which expects a complete experiment configuration) and fill in missing values in some sections with defaults, you will need to pass `partial_run=True` when constructing your experiment object for a partial experiment (this will also avoid cleaning up intermediate tables from the run, equivalent to `cleanup=False`).

Running parts of an experiment is only supported through the Python interface.


### Python

1. `experiment.run()` will run until it no longer has enough configuration to proceed. You will see information in the logs telling you about the steps it was able to perform. You can additionally view the intermediate tables that are built in the database, which are modified with the experiment hash that the experiment calculates, but this will be printed out in the log messages.

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

* triage_metadata.experiments - The experiment configuration, a hash, and some run-invariant details about the configuration
* triage_metadata.experiment_runs - Information about the experiment run that may change from run to run, pertaining to the run environment, status, and results
* triage_metadata.matrices - Each train or test matrix that is built has a row here, with some basic metadata
* triage_metadata.experiment_matrices - A many-to-many table between experiments and matrices. This will have a row if the experiment used the matrix, regardless of whether or not it had to build it
* triage_metadata.models - A model describes a trained classifier; you'll have one row for each trained file that gets saved.
* triage_metadata.experiment_models - A many-to-many table between experiments and models. This will have a row if the experiment used the model, regardless of whether or not it had to build it
* triage_metadata.model_groups - A model groups refers to all models that share parameters like classifier type, hyperparameters, etc, but *have different training windows*. Look at these to see how classifiers perform over different training windows.
* triage_metadata.matrices - Each matrix that was used for training and testing has metadata written about it such as the matrix hash, length, and time configuration.
* triage_metadata.subsets - Each evaluation subset that was used for model scoring has its configuation and a hash written here
* train_results.feature_importances - The sklearn feature importances results for each trained model
* train_results.predictions - Prediction probabilities for train matrix entities generated against trained models
* train_results.prediction_metadata - Metadata about the prediction stage for a model and train matrix, such as tiebreaking configuration
* train_results.evaluations - Metric scores of trained models on the training data.
* test_results.predictions - Prediction probabilities for test matrix entities generated against trained models
* test_results.prediction_metadata - Metadata about the prediction stage for a model and test matrix, such as tiebreaking configuration
* test_results.evaluations - Metric scores of trained models over given testing windows and subsets
* test_results.individual_importances - Individual feature importance scores for test matrix entities.

Here's an example query, which returns the top 10 model groups by precision at the top 100 entities:

```
    select
    	model_groups.model_group_id,
    	model_groups.model_type,
    	model_groups.hyperparameters,
    	max(test_evaluations.value) as max_precision
    from triage_metadata.model_groups
    	join triage_metadata.models using (model_group_id)
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

## Optimizing Experiment Performance

### Profiling an Experiment

Experiment running slowly? Try the `profile` keyword argument, or `--profile` in the command line. This will output a cProfile file to the project path's `profiling_stats` directory.  This is a binary format but can be read with a variety of visualization programs.

[snakeviz](https://jiffyclub.github.io/snakeviz/) - A browser based graphical viewer.
[tuna](https://github.com/nschloe/tuna) - Another browser based graphical viewer
[gprof2dot](https://github.com/jrfonseca/gprof2dot) - A command-line tool to convert files to graphviz format
[pyprof2calltree](https://pypi.org/project/pyprof2calltree/) - A command-line tool to convert files to Valgrind log format, for viewing in established viewers like KCacheGrind

Looking at the profile through a visualization program, you can see which portions of the experiment are taking up the most time. Based on this, you may be able to prioritize changes. For instance, if cohort/label/feature table generation are taking up the bulk of the time, you may add indexes to source tables, or increase the number of database processes. On the other hand, if model training is the culprit, you may temporarily try a smaller grid to get results more quickly.

### materialize_subquery_fromobjs
By default, experiments will inspect the `from_obj` of every feature aggregation to see if it looks like a subquery, create a table out of it if so, index it on the `knowledge_date_column` and `entity_id`, and use that for running feature queries. This can make feature generation go a lot faster if the `from_obj` takes a decent amount of time to run and/or there are a lot of as-of-dates in the experiment. It won't do this for `from_objs` that are just tables, or simple joins (e.g. `entities join events using (entity_id)`) as the existing indexes you have on those tables should work just fine.

You can turn this off if you'd like, which you may want to do if the `from_obj` subqueries return a lot of data and you want to save as much disk space as possible. The option is turned off by passing `materialize_subquery_fromobjs=False` to the Experiment.

### Build Features Independently of Cohort

By default the feature queries generated by your feature configuration on any given date are joined with the cohort table on that date, which means that no features for entities not in the cohort are saved. This is to save time and database disk space when your cohort on any given date is not very large and allow you to iterate on feature building quickly by default. However, this means that anytime you change your cohort, you have to rebuild all of your features. Depending on your experiment setup (for instance, multiple large cohorts that you experiment with), this may be time-consuming. Change this by passing `features_ignore_cohort=True` to the Experiment constructor, or `--save-all-features` to the command-line.


## Experiment Classes

- *SingleThreadedExperiment*: An experiment that performs all tasks serially in a single thread. Good for simple use on small datasets, or for understanding the general flow of data through a pipeline.
- *MultiCoreExperiment*: An experiment that makes use of the pebble library to parallelize various time-consuming steps. Takes an `n_processes` keyword argument to control how many workers to use.
- *RQExperiment*: An experiment that makes use of the python-rq library to enqueue individual tasks onto the default queue, and wait for the jobs to be finished before moving on. python-rq requires Redis and any number of worker processes running the Triage codebase. Triage does not set up any of this needed infrastructure for you. Available through the RQ extra ( `pip install triage[rq]` )
