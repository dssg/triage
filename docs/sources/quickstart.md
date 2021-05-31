# Quickstart guide to using Triage

### 1. Install Triage

Triage can be installed using pip or through python setup.py. It
requires Python 3+ and access to a postgresql database. Ideally you
have full access to a databse so triage can create additional schemas
inside that it needs to store metadata, predictions, and evaluation
metrics.

We also recommend installing triage inside a python virtual
environment for your project so you don't have any conflicts with
other packages installed on the machine. You can use [virutalenv](https://virtualenv.pypa.io/en/latest/) or
[pyenv](https://github.com/pyenv/pyenv-installer/blob/master/README.rst) to do that.

If you use [pyenv](https://github.com/pyenv/pyenv-installer/blob/master/README.rst) (be sure your default python is 3+):
```bash
$ pyenv virtualenv triage-env
$ pyenv activate triage-env
(triage-env) $ pip install triage
```

If you use [virtualenv](https://virtualenv.pypa.io/en/latest/) (be sure your default python is 3+):
```bash
$ virtualenv triage-env
$ . triage-env/bin/activate
(triage-env) $ pip install triage
```

![workflow](dirtyduck/images/quickstart.png "Triage Workflow")

### 2. Make sure your have access to a Postgres database

You'll need to have the servername, databasename, username, and password and put it in a credentials file in Step 5 below.

### 3. Structure your data

The simplest way to start is to structure your data as a series of
events connected to your entity of interest (people, organization,
business, etc.) that take place at a certain time. Each row of the
data will be an **event**. Each event will have some `event_id`, and an
`entity_id` to link it to the entity it happened to, a date, as well as
additional attributes about the event (`type`, for example) and the
entity (`age`, `gender`, `race`, etc.). A sample row might look like:

```
event_id, entity_id, date, event_attribute (type), entity_attribute (age), entity_attribute (gender), ...
121, 19334, 1/1/2013, Placement, 12, Male, ...
```

Triage needs a field named `entity_id` (that needs to be of type
integer) to refer to the primary *entities* of interest in our
project.

### 4. Set up Triage configuration files

The Triage configuration file sets up the modeling process to mirror the
operational scenario the models will be used in. This involves
defining the cohort to train/predict on, the outcome we're predicting,
how far out we're predicting, how often will the model be updated, how
often will the predicted list be used for interventions, what are the
resources available to intervene to define the evaluation metric,
etc.

A lot of details about each section of the configration file can be found [here](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml), but for the moment we'll start with the much simpler configuration file below:

```yaml
config_version: 'v7'

model_comment: 'quickstart_test_run'

temporal_config:
    label_timespans: ['<< YOUR_VALUE_HERE >>']

label_config:
  query: |
    << YOUR_VALUE_HERE >>
  name: 'quickstart_label'

feature_aggregations:
  -
    prefix: 'qstest'
    from_obj: '<< YOUR_VALUE_HERE >>'
    knowledge_date_column: '<< YOUR_VALUE_HERE >>'

    aggregates_imputation:
      count:
        type: 'zero_noflag'

    aggregates:
      -
        quantity:
          total: "*"
        metrics:
          - 'count'

    intervals: ['all']

    groups:
      - 'entity_id'

model_grid_preset:  'quickstart'

scoring:
    testing_metric_groups:
        -
          metrics: [precision@]
          thresholds:
            percentiles: [1]


    training_metric_groups:
      -
          metrics: [precision@]
          thresholds:
            percentiles: [1]
```

Copy that code block into your text editor of choice and save it as something like `quickstart-config.yaml` in your working directory for your project. You'll need to fill out the sections marked `<< YOUR_VALUE_HERE >>` with values appropriate to your project.

The configuration file has a lot of sections. As a first pass, we will
infer a lot of the parameters that are needed in there and use
defaults for others. The primary parameters to specify (for now) are:

1. TIMECHOP config: This sets up temporal parameters for training and
   testing models. The key things to set up here are your prediction
   horizon/timespan (how  far out in the future do you want to
   predict?). For example, if you want to predict an outcome within
   one year, you would set `label_timespans = '12month'`. See
   our [guide to Temporal
   Validation](https://dssg.github.io/triage/experiments/temporal-validation/)

2. LABEL config: This is a `sql` query that defines the outcome of
   interest. 
   
   The query should return a relation containing the columns
   - `entity_id`: each `entity_id` affected by an event within the amount of time specified by `label_timespan` after a given `as_of_date`
   - `outcome`: a binary variable representing the events that happened to each entity, within the period specified by that `as_of_date` and `label_timespan`

   The query is parameterized over `as_of_date`, and `label_timespan`. These parameters are passed to your query as named keywords using the Python's [`str.format()`](https://docs.python.org/3.7/library/stdtypes.html#str.format) method. You can use them in your query by surrounding their keywords with curly braces (as in the example below).
   
   See our
   [guide to Labels](https://dssg.github.io/triage/experiments/cohort-labels/) for a more in-depth discussion of this topic.
   
   **Example Query** 
   
   Given a source table called `semantic.events`, with the following structure:

   |entity_id|event_date|label|
   |-|-|-|
   |135|2014-06-04|1|
   |246|2013-11-05|0|
   |135|2013-04-19|0|
   
   Assuming an early-warning problem, where a client wants to predict the likelihood that each entity experiences at least one positive event (such as a failed inspection) within some period of time, we could use the following label query:

   ```
   select entity_id, max(label) as outcome
   from semantic.events
   where '{as_of_date}'::timestamp <= event_date
         and event_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
   ```

   For each `as_of_date`, this query returns:
   - all `entity_ids` that experienced at least one event (such as an inspection) within the amount of time specified by `label_timespan`
   - a binary variable that equals 1 if an entity experienced at least one positive event (failed inspection), or 0 if all events experienced by the entity had negative results.

3. FEATURE config: This is where we define different aggregate
   features/attributes/variables to be created and used in our machine
   learning models. We need at least one feature specified here. For the purposes of the quickstart, let's just take the count of all events before the modeling date. In the template, you can simply fill in `from_obj` with the `schema.table_name` where your data can be found (but this can also be a more complex query in general) and `knowledge_date_column` with that table's date column.

4. MODEL_GRID_PRESET config: Which models and hyperparameters we want to try in
   this run. We can start with `quickstart` that will run a quick
   model grid to test if everything works.

Additionally, we will need a database credential file that contains the name of the database, server, username, and password to use to connect to it:

```yaml
# Connecting to the database requires a configuration file like this one but
# named database.yaml

host: address.of.database.server
user: user_name
db: database_name
pass: user_password
port: connection_port (often 5432)
```

Copy this into a separate text file, fill in your values and save it as `database.yaml` in the working directory where you'll be running triage. Note, however, that if you have a `DATABASE_URL` environment variable set, triage will use this by default as well.


### 5. Run Triage

An overview of different steps that take place when you run Triage is
[here](https://dssg.github.io/triage/experiments/algorithm/)

For this quickstart, you shouldn't need much free disk space, but note that in general your project path will contain both data matrices and trained model objects, so will need to have ample free space (you can also specify a location in S3 if you don't want to store the files locally).

If you want a bit more detail or documentation, a good overview of running an experiment in triage is [here](https://dssg.github.io/triage/experiments/running/)

#### Using Triage CLI:

1. Validate the configuration files by running:
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run Triage
```
triage experiment config.yaml --project-path '/project_directory'
```

#### Using the Triage python interface:

1. Import packages and load config files:
```py
import yaml
from triage.experiments import SingleThreadedExperiment
from sqlalchemy.engine.url import URL
from triage.util.db import create_engine

with open('config.yaml', 'r') as fin:
    experiment_config = yaml.load(fin)

with open('database.yaml', 'r') as fin:
    db_config = yaml.load(fin)
```

2. Create a database engine and Triage experiment 
```py
# create postgres database url
db_url = URL(
            'postgres',
            host=db_config['host'],
            username=db_config['user'],
            database=db_config['db'],
            password=db_config['pass'],
            port=db_config['port'],
        )

experiment = SingleThreadedExperiment(
    config=experiment_config
    db_engine=create_engine(db_url)
    project_path='/path/to/directory/to/save/data'
)
```

3. Validate your config

```py
experiment.validate()
```

4. Run Triage
```python
experiment.run()
```

### 6. Look at results generated by Triage

Once the feature/cohort/label/matrix building is done and the
experiment has moved onto modeling, check out the
`triage_metadata.models` and `test_results.evaluations` tables as data
starts to come in.

Here are a couple of quick queries to help get you started:

Tables in the `triage_metadata` schema have some general information about
experiments that you've run and the models they created. The `quickstart`
model grid preset should have built 3 models. You can check that with:

```sql
select 
  model_id, model_group_id, model_type 
  from 
      triage_metadata.models;
```

This should give you a result that looks something like:

model_id | model_group_id | model_type
----------|----------------|--------------------------------
1 | 1 | triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression
2 | 2 | sklearn.tree.DecisionTreeClassifier
3 | 3 | sklearn.dummy.DummyClassifier

If you want to see predictions for individual entities, you can check out
`test_results.predictions`, for instance:

```sql
select 
  model_id, entity_id, as_of_date, score, label_value
  from
      test_results.predictions
  limit 5;
```

This will give you something like:

model_id | entity_id |     as_of_date      |  score  | label_value
----------|-----------|---------------------|---------|-------------
1 | 15596 | 2017-09-29 00:00:00 | 0.21884 | 0
2 | 15596 | 2017-09-29 00:00:00 | 0.22831 | 0
3 | 15596 | 2017-09-29 00:00:00 | 0.25195 | 0

Finally, `test_results.evaluations` holds some aggregate information on model
performance:

```sql
select 
  model_id, metric, parameter, stochastic_value
  from
      test_results.evaluations
  order by model_id, metric, parameter;
```

Feel free to explore some of the other tables in these schemas (note that
there's also a `train_results` schema with performance on the training
set as well as feature importances, where defined).

In a more complete modeling run, you could `audition` with jupyter notebooks to help you
select the best-performing model specifications from a wide variety of options (see the [overview of
model selection](https://dssg.github.io/triage/audition/audition_intro/) and [tutorial audition notebook](https://github.com/dssg/triage/blob/master/src/triage/component/audition/Audition_Tutorial.ipynb)) and `postmodeling` to delve deeper into understanding these models (see the [README](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/README.md) and [tutorial postmodeling notebook](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/postmodeling_tutorial.ipynb)).


### 7. Iterate and Explore

Now that you have triage running, [continue onto the suggested project workflow](https://dssg.github.io/triage/triage_project_workflow/) for some tips about how to iterate and tune the pipeline for your project.

Alternatively, if you'd like more of a guided tour with sample data, check out our [dirty duck tutorial](https://dssg.github.io/triage/dirtyduck/).
