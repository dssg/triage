# Running an Experiment

## Prerequisites

To use a Triage experiment, you first need:

- Python 3.5
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk (or S3) to store the needed matrices and models for your experiment
- An experiment definition (see [Defining an Experiment](defining.md))


## Experiment Run Example

The basic execution of an experiment looks something like the following:

```
	SingleThreadedExperiment(
		config=experiment_config,
		db_engine=sqlalchemy.create_engine(...),
		model_storage_class=FSModelStorageEngine,
		project_path='/path/to/directory/to/save/data'
	).run()
```

These lines are a bit dense: what is happening here?

- `SingleThreadedExperiment`:  There are different Experiment classes available in `triage.experiments` to use, and they each represent a different way of executing the experiment, which we'll talk about in more detail later. The simplest (but slowest) is the SingleThreadedExperiment.
- `config=experiment_config`: The bulk of the work needed in designing an experiment will be in creating this experiment configuration. An up-to-date example is at [example_experiment_config.yaml](example_experiment_config.yaml); more detailed instructions on each section are located in the example file. Generally these would be easiest to store as a file (or multiple files that you construct together) like that YAML file, but the configuration is passed in dict format to the Experiment constructor and you can store it however you wish.
- `db_engine=sqlalchemy.create_engine(...)`: A SQLAlchemy database engine. This will be used both for querying your source tables and writing results metadata.
- `model_storage_class=FSModelStorageEngine`: The path to a model storage engine class. The library that triage uses for model training and evaluation, [catwalk](https://github.com/dssg/catwalk), provides multiple classes that handle storing trained models in different mediums, such as on the local filesystem or Amazon S3. We recommend starting with the `catwalk.storage.FSModelStorageEngine` to save models on the local filesystem.
- `project_path='/path/to/directory/to/save/data'`: The path to where you would like to store design matrices and trained models.

With that in mind, a more full version of the experiment run script might look like this:

```
import sqlalchemy
import yaml

from catwalk.storage import FSModelStorageEngine
from triage.experiments import SingleThreadedExperiment

with open('my_experiment_config.yaml') as f:
	experiment_config = yaml.load(f)
with open('my_database_creds') as f:
	db_connection_string = yaml.load(f)['db_connection_string']

experiment = SingleThreadedExperiment(
	config=experiment_config,
	db_engine=sqlalchemy.create_engine(db_connection_string),
	model_storage_class=FSModelStorageEngine,
	project_path='/home/research/myproject'
)

experiment.run()
```


## Evaluating results of an Experiment

After the experiment run, a results schema will be created and populated in the configured database with the following tables:

* experiments - The experiment configuration and a hash
* models - A model describes a trained classifier; you'll have one row for each trained file that gets saved.
* model_groups - A model groups refers to all models that share parameters like classifier type, hyperparameters, etc, but *have different training windows*. Look at these to see how classifiers perform over different training windows.
* feature_importances - The sklearn feature importances results for each trained model
* predictions - Prediction probabilities for entities generated against trained models
* evaluations - Metric scores of trained models over given testing windows

Here's an example query, which returns the top 10 model groups by precision at the top 100 entities:
```
select
	model_groups.model_group_id,
	model_groups.model_type,
	model_groups.model_parameters,
	max(evaluations.value) as max_precision
from model_groups
	join models using (model_group_id)
	join evaluations using (model_id)
where
	metric = 'precision@'
	and parameter = '100_abs'
group by 1,2,3
order by 4 desc
limit 10
```

The resulting schema is also readable by [Tyra](https://github.com/tyra), our model evaluation webapp.


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

