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

    ./tutorial .sh up

 from you command line. This will start the database.

### 1. Install Triage: Check!

We also containerized `triage`, so, in this tutorial it is already
installed for you! Just run

    ./tutorial.sh bastion

The prompt in your command line should change to something like

    [dirtyduck@bastion$:/triage]

Type `triage`, if no error. You completed this step!

Now you have `triage` installed, with all its power at the point of
your fingers.

### 2. Structure your data: Events (and entities)

As mentioned in the [quickstart
workflow](../quickstart.md#2-structure-your-data), *at least* you need
one table that contains *events*, i.e. something that happened to your
entities of interest somewhere at sometime. So you need *at least*
three columns in your data: `event_id`, `event_id`, `date` (and
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
`entity_id` (which links  to the entity it happened to) , a `date`,
(when it happened) as well a location  (the `zip_code` column). The
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

### 3. Set up Dirty duck's triage configuration file

The configuration file sets up the modeling process to mirror the
operational scenario the models will be used in. This involved
defining the cohort to train/predict on, the outcome we're predicting,
how  far out we're predicting, how often will the model be updated,
how often will the predicted list be used for interventions, what are
the resources available to intervene to define the evaluation metric,  etc.

Here's the [sample configuration
file](dirtyduck/experiments/dirty-duckling.yaml) called `dirty-duckling.yaml`


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
            percentiles: [1]


    training_metric_groups:
      -
          metrics: [precision@]
          thresholds:
            percentiles: [1]

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
database configuation file](database.yaml) or you can setup an
environment variable named `$DATABASE_URL`, this is the approach taken
in the dirtyduck tutorial, its value inside `bastion` is

       postgresql://food_user:some_password@food_db/food

For the quick explanation of the sections check the[quickstart
workflow guide](../quickstart.md#3-set-up-triage-configuration-files). For a
detailed explanation about each section of the configuration file look
[here](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml)



### 4. Run triage

Now we are ready for run something! First we will validate the
configuration files by running:

```shell
triage experiment experiments/dirty-duckling.yaml --validate-only
```

If everything was OK (it should!), you can run the experiment with:

```bash
triage experiment experiments/dirty-duckling.yaml
```
That's it! If you see this message in your screen:

     INFO:root:Experiment complete
        INFO:root:All models that were supposed to be trained were trained. Awesome!
        INFO:root:All matrices that were supposed to be build were built. Awesome!

it would mean that `triage` actually built (in this order) cohort
(table `cohort_all_entities...`),
labels (table `labels_failed_inspections...`), features (schema
`features`), matrices (table `model_metdata.matrices` and folder
`matrices`), models (tables `model_metadata.models` and
`model_metadata.model_groups`; folder `trained_models`), predictions
(table `test_results.predictions`)
and evaluations (table `test_results.evaluations`).

### 5. Look at results of your duckling!

You can check to the tables in the schemas `model_metadata` and
`test_results`. There you will find a lot of information related to
the performance of your models. With all that data you could (*should*) do model
selection, postmodeling, bias audit, etc.

`triage` provides tools for doing all of that, but we should stop this
little experiment. If you successfully arrive to this point, now you
are all set to do your own modeling, but if you want to go deeper in
all the things that `triage` could do for you, continue reading:


- [Take a deeper look at triage through this example](problem_description.md)

- [Get started with your own project and data](../quickstart.md)
