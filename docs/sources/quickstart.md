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

### 2. Structure your data

The simplest way to start is to structure your data as a series of
events connected to your entity of interest (people, organization,
business, etc.) that take place at a certain time. Each row of the
data will be an **event**. Each event will have some `event_id`, and an
`entity_id` to link it to the entity it happened to, a date, as well as
additional attributes about the event (type for example) and the
entity (`age`, `gender`, `race`, etc.). A sample row might look like:

```
event_id, entity_id, date, event_attribute (type), entity_attribute (age), entity_attribute (gender), ...
121, 19334, 1/1/2013, Placement, 12, Male, ...
```

Triage needs a field named `entity_id` (that needs to be of type
integer) to refer to the primary *entities* of interest in our
project.

### 3. Set up Triage configuration files

The configuration file sets up the modeling process to mirror the
operational scenario the models will be used in. This involved
defining the cohort to train/predict on, the outcome we're predicting,
how far out we're predicting, how often will the model be updated, how
often will the predicted list be used for interventions, what are the
resources available to intervene to define the evaluation metric,
etc.

A lot of details about each section of the configration file can be found [here](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml), but for the moment we'll start with the much simplier configuration file below:

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

2. LABEL config: This is a `sql` query that defines what the outcome of
   interest is. The query must return two columns: `entity_id` (an integer) and
   `outcome` (with integer label values of `0` and `1`), based on a given `as_of_date` and `label_timespan` (you can use these parameters in your query by surrounding them with curly braces as in the example below). See our
   [guide to Labels](https://dssg.github.io/triage/experiments/cohort-labels/). For example, if your data was in a table called `semantic.events` containing columns `entity_id`, `event_date`, and `label`, this query could simply be:
   ```
   select entity_id, max(label) as outcome
   from semantic.events
   where '{as_of_date}'::timestamp <= event_date
         and event_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
   ```

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


### 4. Run Triage

An overview of different steps that take place when you run Triage is
[here](https://dssg.github.io/triage/experiments/algorithm/)

1. Validate the configuration files by running:
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run triage
```
triage experiment config.yaml --project-path '/project_directory'
```

For this quickstart, you shouldn't need much free disk space, but note that in general your project path will contain both data matrices and trained model objects, so will need to have ample free space (you can also specify a location in S3 if you don't want to store the files locally).

If you want a bit more detail or documentation, a good overview of running an experiment in triage is [here](https://dssg.github.io/triage/experiments/running/).


### 5. Look at results generated by Triage

Once the feature/cohor/label/matrix building is done and the
experiment has moved onto modeling, check out the
`model_metadata.models` and `test_results.evaluations` tables as data
starts to come in.

We can either look at results directly in the database (`test_results`
schema) or use `audition` by installing jupyter notebook. [Overview of
model
selection](https://dssg.github.io/triage/dirtyduck/docs/audition/)


### 6. Iterate and Explore

Now that you have triage running, [continue onto the suggested project workflow](https://dssg.github.io/triage/triage_project_workflow/) for some tips about how to iterate and tune the pipeline for your project.

Alternatively, if you'd like more of a guided tour with sample data, check out our [dirty duck tutorial](https://dssg.github.io/triage/dirtyduck/).
