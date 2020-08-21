# Dirty duckling: the quick start guide

![workflow](images/quickstart.png "Triage Workflow")


This *quickstart* guide follows the workflow explained
[here](../quickstart.md). The goal is to show you an instance of that
workflow using the [Chicago Food Inspections
dataset](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5)
data source.

We packed this a sample of Chicago Food Inspections data source as
part of the dirty duck tutorial.  Just run in the folder that contains
the `triage` local repository:

    ./tutorial.sh up

 from you command line. This will start the database.

### 1. Install Triage: Check!

We also containerized `triage`, so, in this tutorial it is already
installed for you! Just run

    ./tutorial.sh bastion

The prompt in your command line should change to something like

    [triage@dirtyduck$:/dirtyduck]#

Type `triage`, if no error. You completed this step!

Now you have `triage` installed, with all its power at the point of
your fingers.

### 2. Structure your data: Events (and entities)

As mentioned in the [quickstart
workflow](../quickstart.md#2-structure-your-data), *at least* you need
one table that contains *events*, i.e. something that happened to your
entities of interest somewhere at sometime. So you need *at least*
three columns in your data: `entity_id`, `event_id`, `date` (and
`location` if you have it would be a nice addition).

In dirtyduck, we provide you with two tables: `semantic.entities` and
`semantic.events`. The latter is the required minimum. We added the
`semantic.entities` table as a good practice.

This is the  simplest way to structure your data: as a series of
events connected to your entity of interest (people, organization,
business, etc.) that take place at a certain time. Each row of the
data will be an event.

For this quickstart tutorial, you don't need to interact manually with
the database, but, if you are curious you can peek inside it, and
verify how the `events` table look like.

Inside `bastion` you can connect to the database typing

    psql $DATABASE_URL

This will change the prompt one more time to

    food=#

Now, type (or copy-paste) the following

```sql
select
  event_id
  entity_id,
  date,
  zip_code,
  type
  from
      semantic.events
 where random() < 0.001
 limit 5;
```

 entity_id |    date    | zip_code |   type
----------|------------|----------|-----------
   1092838 | 2014-02-27 | 60657    | license
   1325036 | 2014-05-19 | 60612    | canvass
   1385431 | 2014-06-25 | 60651    | complaint
   1395315 | 2014-01-08 | 60707    | canvass
   1395916 | 2014-02-03 | 60641    | canvass

Each row in this table is an event with `event_id` and
`entity_id` (which links to the entity it happened to), a `date`
(when it happened), as well a location (the `zip_code` column). The
event will have attributes that describe it in its particularity, in
this case we are just showing one of those attributes: the type of the
inspection (`type`)

And, if you also want to see the *entities* in your data

```sql
select
  entity_id, license_num, facility, facility_type, activity_period
  from
      semantic.entities
 where random() < 0.001 limit 5;
```

entity_id | license_num |        facility        | facility_type | activity_period
 -----------|-------------|------------------------|---------------|-----------------
 2218 |     1223576 | loretto hospital       | hospital      | [2014-02-27,)
 2353 |     1804587 | subway                 | restaurant    |[2014-03-05,)
 636 |     2002788 | duck walk                | restaurant    | [2014-01-17,2016-02-29)
 3748 |     1904141 | zaragoza restaurant    | restaurant    | [2014-04-03,)
 5118 |     2224978 | saint cajetan          | school        | [2014-05-06,)

Triage needs a field named `entity_id` (that needs to be of type
integer) to refer to the primary *entities* of interest in our
project.

When you're done exploring the database, you can exit the postgres command line interface by typing `\q`

### 3. Set up Dirty duck's triage configuration file

The configuration file sets up the modeling process to mirror the
operational scenario the models will be used in. This involves
defining the cohort to train/predict on, the outcome we're predicting,
how far in the future we're predicting, how often will the model be updated,
how often will the predicted list be used for interventions, what are
the resources available to intervene to define the evaluation metric,  etc.

Here's the [sample configuration
file](https://github.com/dssg/triage/blob/master/example/dirtyduck/experiments/dirty-duckling.yaml) called `dirty-duckling.yaml`

If you wish, you can check the content of the file with `cat
experiments/dirty-ducking.yaml`

```yaml
config_version: 'v7'

model_comment: 'dirtyduck-quickstart'

temporal_config:
    label_timespans: ['3months']

label_config:
  query: |
    select
    entity_id,
    bool_or(result = 'fail')::integer as outcome
    from semantic.events
    where '{as_of_date}'::timestamp <= date
    and date < '{as_of_date}'::timestamp + interval '{label_timespan}'
    group by entity_id
  name: 'failed_inspections'

feature_aggregations:
  -
    prefix: 'inspections'
    from_obj: 'semantic.events'
    knowledge_date_column: 'date'

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
            percentiles: [10]


    training_metric_groups:
      -
          metrics: [precision@]
          thresholds:
            percentiles: [10]

```

This is the *minimum* configuration file, and it still has a lot of
sections (ML is a complex business!).

!!! warning
    If you use the minimum configuration file several parameters will fill
    up using defaults. Most of the time those defaults are not the values
    that your modeling of the problem needs! Please check
    [here](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml)
    to see which values are being used and act accordingly.

`triage` uses/needs a *data connection* in order to work. The
connection will be created using the database credentials information
(name of the database, server, username, and password).

You could use a database configuration file [here's an example
database configuation file](https://github.com/dssg/triage/blob/master/example/config/database.yaml) or you can setup an
environment variable named `$DATABASE_URL`, this is the approach taken
in the dirtyduck tutorial, its value inside `bastion` is

       postgresql://food_user:some_password@food_db/food

For the quick explanation of the sections check the [quickstart
workflow guide](../quickstart.md#3-set-up-triage-configuration-files). For a
detailed explanation about each section of the configuration file look
[here](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml)



### 4. Run triage

Now we are ready for run something! First we will validate the
configuration files by running:

```shell
triage experiment experiments/dirty-duckling.yaml --validate-only
```

If everything was OK (it should!), you will see this in your screen:

```bash
2020-08-20 16:55:34 - SUCCESS Experiment validation ran to completion with no errors
2020-08-20 16:55:34 - SUCCESS Experiment (a336de4800cec8964569d051dc56f85d)'s configuration file is OK!

```

Now you can run the experiment with:

```bash
triage experiment experiments/dirty-duckling.yaml
```
That's it! If you see this message in your screen:

     2020-08-20 16:56:56 - SUCCESS Training, testing and evaluatiog models completed
     2020-08-20 16:56:56 - SUCCESS All matrices that were supposed to be build were built. Awesome!
     2020-08-20 16:56:56 - SUCCESS All models that were supposed to be trained were trained. Awesome!
     2020-08-20 16:56:56 - SUCCESS Experiment (a336de4800cec8964569d051dc56f85d) ran through completion


it would mean that `triage` actually built (in this order) cohort
(table `cohort_all_entities...`),
labels (table `labels_failed_inspections...`), features (schema
`features`), matrices (table `model_metdata.matrices` and folder
`matrices`), models (tables `triage_metadata.models` and
`triage_metadata.model_groups`; folder `trained_models`), predictions
(table `test_results.predictions`)
and evaluations (table `test_results.evaluations`).

### 5. Look at results of your duckling!

Next, let's quickly check the tables in the schemas `triage_metadata` and
`test_results` to make sure everything worked. There you will find a lot
of information related to the performance of your models.

Still connected to the `bastion` docker container, you can connect to the
database by typing:

    psql $DATABASE_URL

Again, you should see the postgreSQL prompt:

    food=#

Tables in the `triage_metadata` schema have some general information about
experiments that you've run and the models they created. The `quickstart`
model grid preset should have built 3 models. Let's check with:

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
  where entity_id = 15596
  order by model_id;
```

This will give you something like:

model_id | entity_id |     as_of_date      |  score  | label_value
----------|-----------|---------------------|---------|-------------
1 | 15596 | 2017-09-29 00:00:00 | 0.21884 | 0
2 | 15596 | 2017-09-29 00:00:00 | 0.22831 | 0
3 | 15596 | 2017-09-29 00:00:00 | 0.25195 | 0

Finally, `test_results.evaluations` holds some aggregate information on model
performance. In our config above, we only focused on precision in the top
ten percent, so let's see how the models are doing based on this:

```sql
select
  model_id, metric, parameter,
  round(stochastic_value,3) as stochatic_value
  from
      test_results.evaluations
  where metric = 'precision@'
        and parameter='10_pct'
  order by model_id;
```

model_id |   metric   | parameter |  stochastic_value
----------|------------|-----------|--------------------
1 | precision@ | 10_pct     | 0.287
2 | precision@ | 10_pct     | 0.292
3 | precision@ | 10_pct     | 0.237

Not great! But then again, these were just a couple of overly simple model
specifications to get things up and running...

Feel free to explore some of the other tables in these schemas (note that
there's also a `train_results` schema with performance on the training
set as well as feature importances, where defined). When you're done
exploring the database, you can exit the postgres command line interface
by typing `\q`

With a real modeling run you could (*should*) do model selection, postmodeling,
bias audit, etc. `triage` provides tools for doing all of that, but we
the purpose of this little experiment was just to get things up and running.
If you have successfully arrived to this point, you are all set to do your own
modeling ([here's a good place to start](../quickstart.md)), but if you want to
go deeper in this example and learn about these other `triage` functions,
[continue reading our in-depth tutorial](problem_description.md).


### 6. What's next?

### [Take a deeper look at triage through this example](problem_description.md)

### [Get started with your own project and data](../quickstart.md)
