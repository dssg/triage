# Cohort and Label Deep Dive

This document is intended at providing a deep dive into the concepts of cohorts and labels as they apply to Triage. For context, reading the [Triage section of the Dirty Duck tutorial](https://dssg.github.io/dirtyduck/#sec-4-2) may be helpful before reading this document.

## Temporal Validation Refresher

Triage uses temporal validation to select models because the real-world problems that Triage is built for tend to evolve or change over time.  Picking a date range to train on and a date range afterwards to test on ensures that we don't leak data from the future into our models that wouldn't be available in a real-world deployment scenario. Because of this, we often talk in Triage about the *as-of-date*: all models trained by Triage are associated with an *as-of-date*, which means that all the data that goes into the model **is only included if it was known about before that date**. The matrix used to train the model may have multiple *as-of-dates*, and the most recent is referred to as the model's *as-of-date* so it's easy to see the cutoff date for data included in the model. For more on temporal validation, see the [relevant section in Dirty Duck](https://dssg.github.io/dirtyduck/#sec-4-2-2-1).

## What are Cohorts and Labels in Triage?

This document assumes that the reader is familiar with the concept of a machine learning target variable and will focus on explaining what is unique to Triage.

**A cohort is the population used used for modeling on a given as-of-date**. This is expressed as a list of *entities*. An entity is simply the object of prediction, such as a facility to inspect or a patient coming in for a visit. Early warning systems tend to include their entire population (or at least a large subset of it) in the cohort at any given date, while appointment-based problems may only include in a date's cohort the people who are scheduled for an appointment on that date.

**A label is the binary target variable for a member of the cohort at a given as-of-date and a given label timespan.** For instance, in an inspection prioritization problem the question being asked may be 'what facilities are at high risk of having a failed inspection in the next 6 months?' For this problem, the `label_timespan` is 6 months. There may be multiple label timespans tested in the same experiment, in which case there could be multiple labels for an entity and date. In addition, multiple label definitions are often tested against each other, such as "any inspection failures" vs "inspection failures with serious issues".

Both labels and cohorts are defined in Triage's experiment configuration using SQL queries, with the variables (`as_of_date`, `label_timespan`) given as placeholders. This allows the definitions to be given in a concise manner while allowing the temporal configuration defined elsewhere in the experiment to produce the actual list of dates and timespans that are calculated during the experiment.


## Cohort Definition and Examples

The cohort is configured with a query that returns a unique list of `entity_id`s given an `as_of_date`, and it runs the query for each `as_of_date` that is produced by your temporal config. You tell Triage where to place each `as_of_date` with a placeholder surrounded by brackets: `{as_of_date}`.

### Note 1

The as_of_date is parsed as a timestamp in the database, which Postgres defaults to **midnight at the beginning of the date in question**. It's important to consider how this is used for feature generation. Features are only included if they are known about **before this timestamp**. So features will be only included for an as_of_date if they are known about **before that as_of_date**. If you want to work around this (e.g for visit-level problems in which you want to intake data **on the day of the visit and make predictions using that data the same day**), you can move your cohort up a day. The time splitting in Triage is designed for day granularity so approaches to train up to a specific hour and test at another hour of the same day are not supported.


### Note 2

Triage expects all entity ids to be integers.

### Note 3

Triage expects the cohort to be a unique list of entity ids. Throughout the cohort example queries you will see `distinct(entity_id)` used to ensure this.

### Example: Inspections
Let's say I am prioritizing the inspection of food service facilities such as restaurants, caterers or grocery stores. One simple definition of a cohort for facility inspection would be to include *any facilities that have active permits in the last year* in the cohort. Assume that these permits are contained in a table, named `permits`, with the facility's id, a start date, and an end date of the permit.

#### Inspections Cohort Source Table

entity_id | start_date | end_date
------------ | ------------- | ------------
25 | 2016-01-01  | 2016-02-01
44 | 2016-01-01  | 2016-02-01
25 | 2016-02-01  | 2016-03-01

Triage expects the cohort query passed to it to return a unique list of `entity_id`s given an `as_of_date`, and it runs the query for each `as_of_date` that is produced by your temporal config. You tell Triage where to place each `as_of_date` with a placeholder surrounded by brackets: `{as_of_date}`. An example query that implements the 'past year' definition would be:

`select distinct(entity_id) from permits where tsrange(start_date, end_date, '[]') @> {as_of_date}`

- Running this query using the `as_of_date` '2017-01-15' would return both entity ids 25 and 44.
- Running it with '2017-02-15' would return only entity id 25.
- Running it with '2017-03-15' would return no rows.

#### Inspections Cohort Config

The way this looks in an Experiment configuration YAML is as follows:

```
cohort_config:
  query: |
    select distinct(entity_id)
    from permits
    where
    tsrange(start_time, end_time, '[]') @> {as_of_date}
  name: 'permits_in_last_year'
```

The `name` key is optional. Part of its purpose is to help you organize different cohorts in your configuration, but it is also included in each matrix's metadata file to help you keep them straight afterwards.


### Example: Early Intervention

An example of an early intervention system is identifying people at risk of recidivism so they can receive extra support to encourage positive outcomes. 


This example defines the cohort as everybody who has been released from jail within the last three years. It does this by querying an events table for events of type 'release'.

#### Early Intervention Cohort Source Table

entity_id | event_type | knowledge_date
------------ | ------------- | ------------
25 | booking  | 2016-02-01
44 | booking  | 2016-02-01
25 | release  | 2016-03-01


#### Early Intervention Cohort Config
```
cohort_config:
    query: |
        SELECT distinct(entity_id)
          FROM events
         WHERE event_type = 'release'
           AND knowledge_date <@ daterange(('{as_of_date}'::date - '3 years'::interval)::date, '{as_of_date}'::date)
    name: 'booking_last_3_years'
```

### Example: Visits

Another problem type we may want to model is visit/appointment level modeling. An example would be a health clinic that wants to figure out which patients on a given day who are most at risk for developing diabetes within some time period but don't currently have it.

#### Visits Cohort Source Tables

Here we actually define two tables: an appointments table that contains the appointment schedule, and a diabetes diagnoses table that contains positive diabetes diagnoses.

`appointments`

entity_id | appointment_date
------------ | -------------
25 | 2016-02-01
44 | 2016-02-01
25 | 2016-03-01

`diabetes_diagnoses`

entity_id | diagnosis_date
------------ | -------------
44 | 2015-02-01
86 | 2012-06-01

#### Visits Cohort Config

The cohort config here queries the visits table for the next day, and excludes those who have a diabetes diagnosis at some point in the past. There's a twist: a day is subtracted from the as-of-date. Why? We may be collecting useful data during the appointment about whether or not they will develop diabetes, and we may want to use this data as features. Because the as-of-date refers to the timestamp at the beginning of the day ([see note 1](#note-1)), if the as-of-date and appointment date match up exactly we won't be able to use those features. So, appointments show up in the next day's as-of-date.

Whether or not this is correct depends on the feasability of generating a prediction during the visit to use this data, which depends on the deployment plans for the system.  If data entry and prediction can only happen nightly, you can't expect to use data from the visit in features and would change the as-of-date to match the appointment_date.

```
cohort_config:
  query: |
    select distinct(entity_id)
    from appointments
    where appointment_date = ('{as_of_date}'::date - interval '1 days')::date
      and not exists(
          select entity_id
          from diabetes_diagnoses
          where entity_id = appointments.entity_id
            and as_of_date < '{as_of_date}'
          group by entity_id)
    group by entity_id
    name: 'visit_day_no_previous_diabetes'
```

### Testing Cohort Configuration

If you want to test out a cohort query without running an entire experiment, there are a few ways, and the easiest way depends on how much of the rest of the experiment you have configured.

Option 1: **You have not started writing an experiment config file yet**. If you just want to test your query with a hardcoded list of dates as Triage does it (including as-of-date interpolation), you can instantiate the `EntityDateTableGenerator` with the query and run it for those dates. This skips any temporal config, so you don't have to worry about temporal config:

```python
from triage.component.architect.entity_date_table_generators import EntityDateTableGenerator
from triage import create_engine
from datetime import datetime

EntityDateTableGenerator(
    query="select entity_id from permits where tsrange(start_time, end_time, '[]') @> {as_of_date}",
    db_engine=create_engine(...),
    entity_date_table_name="my_test_cohort_table"
).generate_entity_date_table([datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2016, 3, 1)])
```

Running this will generate a table with the name you gave it (`my_test_cohort_table`), populated with the cohort for that list of dates. You can inspect this table in your SQL browser of choice.

Option 2: **You have an experiment config file that includes temporal config, and want to look at the cohort in isolation in the database**. If you want to actually create the cohort for each date that results from your temporal config, you can go as far as instantiating an Experiment and telling it to generate the cohort.

```python
from triage.experiments import SingleThreadedExperiment
from triage import create_engine
import yaml

with open('<your_experiment_config.yaml>') as fd:
    experiment_config = yaml.full_load(fd)
experiment = SingleThreadedExperiment(
    experiment_config=experiment_config,
    db_engine=create_engine(...),
    project_path='./'
)
experiment.generate_cohort()
print(experiment.cohort_table_name)
```

This will generate the entire cohort needed for your experiment. The table name is autogenerated by the Experiment, and you can retrieve it using the `cohort_table_name` attribute of the Experiment. Here, as in option 1, you can look at the data in your SQL browser of choice.

These options should be enough to test your cohort in isolation. How the cohort shows up in matrices is also dependent on its interaction with the labels, and later we'll show how to test that.

## Label Definition and Examples

The labels table works similarly to the cohort table: you give it a query with a placeholder for an as-of-date. However, the label query has one more dependency: a `label timespan` For instance, if you are inspecting buildings for violations, a label timespan of 6 months translates into a label of 'will this building have a violation in the next 6 months?'.  These label timespans are generated by your temporal configuration as well and you may have multiple in a single experiment, so what you send Triage in your label query is also a placeholder.

**Note: The label query is expected by Triage to return only one row per entity id for a given as-of-date/label timespan combination.**


### Missing Labels

Since the cohort has its own definition query, separate from the label query, we have to consider the possibility that not every entity in the cohort is present in the label query, and how to deal with these missing labels.  The label value in the train matrix in these cases is controlled by a flag in the label config: `include_missing_labels_in_train_as`. 

- If you omit the flag, they show up as missing. This is common for inspections problems, wherein you really don't know a suitable label. The facility wasn't inspected, so you really don't know what the label is. This makes evaluation a bit more complicated, as some of the facilities with high risk scores may have no labels. But this is a common tradeoff in inspections problems.
- If you set it to True, that means that all of the rows have positive label. What does this mean? It depends on what exactly your label query is, but a common use would be to model early warning problems of dropouts, in which the *absence* of an event (e.g. a school enrollment event) is the positive label.
- If you set it to False, that means that all of these rows have a negative label. A common use for this would be in early warning problems of adverse events, in which the *presence* of an event (e.g. excessive use of force by a police officer) is the positive label.


### Example: Inspections

#### Inspections Label Source Table

To translate this into our restaurant example above, consider a source table named 'inspections' that contains information about inspections. A simplified version of this table may look like:


entity_id | date | result
------------ | ------------- | ------------
25 | 2016-01-05  | pass
44 | 2016-01-04  | fail
25 | 2016-02-04  | fail


The entity id is the same as the cohort above: it identifies the restaurant. The date is just the date that the inspection happened, and the result is a string 'pass'/'fail' stating whether or not the restaurant passed the inspection. 


#### Inspections Label Config

In constructing the label query, we have to consider the note above that we want to return only one row for a given entity id. The easiest way to do this, given that this query is run per as-of-date, is to group by the entity id and aggregate all the matched events somehow. In this case, a sensible definition is that we want any failed inspections to trigger a positive label. So if there is one pass and one fail that falls under the label timespan , the label should be True. `bool_or` is a handy Postgres aggregation function that does this.

A query to find any failed inspections would be written in an experiment YAML config as follows:

```yaml
label_config:
  query: |
    select
    entity_id,
    bool_or(result = 'fail')::integer as outcome
    from inspections
    where '{as_of_date}'::timestamp <= date
    and date < '{as_of_date}'::timestamp + interval '{label_timespan}'
    group by entity_id
  name: 'failed_inspection'
```

### Example: Early Intervention

#### Early Intervention Label Source Table

We reuse the [generic events table](#early-intervention-cohort-source-table) used in the early intervention cohort section.

#### Early Intervention Label Config

We would like to assign a `True` label to everybody who is booked into jail within the label timespan. Note the `include_missing_labels_in_train_as` value: `False`. Anybody who does not show up in this query can be assumed to not have been booked into jail, so they can be assigned a `False` label.


```
label_config:
    query: |
        SELECT entity_id,
               bool_or(CASE WHEN event_type = 'booking' THEN TRUE END)::integer AS outcome
          FROM events
         WHERE knowledge_date <@ daterange('{as_of_date}'::date, ('{as_of_date}'::date + interval '{label_timespan}')::date)
         GROUP BY entity_id
    include_missing_labels_in_train_as: False
    name: 'booking'
```

### Example: Visits

#### Visits Label Source Table

We reuse the [diabetes_diagnoses table](#visits-cohort-source-tables) from the cohort section.

#### Visits Label Config

We would like to identify people who are diagnosed with diabetes within a certain `label_timespan` after the given `as-of-date`. Note that `include_missing_labels_in_train_as` is False here as well. Any diagnoses would show up here, so the lack of any results from this query would remove all ambiguity.

```
label_config:
  query: |
    select entity_id, 1 as outcome
    from diabetes_diagnoses
    where as_of_date <@ daterange('{as_of_date}' :: date, ('{as_of_date}' :: date + interval '{label_timespan}') :: date)
    group by entity_id
  include_missing_labels_in_train_as: False
  name: 'diabetes'
```

Note: If you broadened the scope of this diabetes problem to concern not just diabetes diagnoses but having diabetes in general, and you had access to both positive and negative diabetes tests, you might avoid setting `include_missing_labels_in_train_as`, similar to the inspections problem, to more completely take into account the possibility that a person may or may not have diabetes.

### Testing Label Configuration

If you want to test out a label query without running a whole experiment, you can test it out similarly to the cohort section above.

Option 1: **You have not started writing an experiment config file yet**. If you just want to test your label query with a hardcoded list of dates as Triage does it (including as-of-date interpolation), you can instantiate the `LabelGenerator` with the query and run it for those dates. This skips any temporal config, so you don't have to worry about temporal config:

```python
from triage.component.architect.label_generators import LabelGenerator
from triage import create_engine
from datetime import datetime

LabelGenerator(
    query="select entity_id, bool_or(result='fail')::integer as outcome from inspections where '{as_of_date}'::timestamp <= date and date < '{as_of_date}'::timestamp + interval '{label_timespan}' group by entity_id"
    db_engine=create_engine(...),
).generate_all_labels(
    labels_table='test_labels',
    as_of_dates=[datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2016, 3, 1)],
    label_timespans=['3 month'],
)
```

Running this will generate a table with the name you gave it (`test_labels`), populated with the labels for that list of dates. You can inspect this table in your SQL browser of choice.

Option 2: **You have an experiment config file that includes temporal config, and want to look at the labels in isolation in the database**. If you want to actually create the labels for each date that results from your temporal config, you can go as far as instantiating an Experiment and telling it to generate the labels.

```python
from triage.experiments import SingleThreadedExperiment
from triage import create_engine
import yaml

with open('<your_experiment_config.yaml>') as fd:
    experiment_config = yaml.full_load(fd)
experiment = SingleThreadedExperiment(
    experiment_config=experiment_config,
    db_engine=create_engine(...),
    project_path='./'
)
experiment.generate_labels()
print(experiment.labels_table_name)
```

This will generate the labels for each as-of-date in your experiment. The table name is autogenerated by the Experiment, and you can retrieve it using the `labels_table_name` attribute of the Experiment. Here, as in option 1, you can look at the data in your SQL browser of choice.

These options should be enough to test your labels in isolation. How the labels shows up in matrices is also dependent on its interaction with the cohort, and later we'll show how to test that.


## Combining Cohorts and Labels to make Matrices
Looking at the cohort and labels tables in isolation doesn't quite get you the whole picture. They are combined with features to make matrices, and some of the functionality (e.g. `include_missing_labels_in_train_as`) isn't applied until the matrices are made for performance/database disk space purposes.

How does this work? Let's look at some example cohort and label tables.

### Cohort

entity_id | as_of_date
------------ | -------------
25 | 2016-01-01
44 | 2016-01-01
25 | 2016-02-01
44 | 2016-02-01
25 | 2016-03-01
44 | 2016-03-01
60 | 2016-03-01

### Label
entity_id | as_of_date | label
------------ | ------------- | ------------- 
25 | 2016-01-01 | True
25 | 2016-02-01 | False
44 | 2016-02-01 | True
25 | 2016-03-01 | False
44 | 2016-03-01 | True
60 | 2016-03-01 | True

Above we observe three total cohorts, on `2016-01-01`, `2016-02-01`, and `2016-03-01`. The first two cohorts have two entities each and the last one has a new third entity. For the first cohort, only one of the entities has an explicitly defined label (meaning the label query didn't return anything for them on that date).

For simplicity's sake, we are going to assume only one matrix is created that includes all of these cohorts. Depending on the experiment's temporal configuration, there may be one, many, or all dates in a matrix, but the details here are outside of the scope of this document.

In general, the index of the matrix is created using a left join in SQL: The cohort table is the left table, and the labels table is the right table, and they are joined on entity id/as of date. So all of the rows that are in the cohort but not the labels table (in this case, just entity 44/date 2016-01-01) will initially have a null label.

The final contents of the matrix, however, depend on the `include_missing_labels_in_train_as` setting.

### Inspections-Style (preserve missing labels as null)

If `include_missing_labels_in_train_as` is not set, Triage treats it as a truly missing label. The final matrix will look like:

entity_id | as_of_date | ...features... | label
------------ | ------------- | ------------- | ------------- 
25 | 2016-01-01 | ... | True
44 | 2016-01-01 | ... | **null**
25 | 2016-02-01 | ... | False
44 | 2016-02-01 | ... | True
25 | 2016-03-01 | ... | False
44 | 2016-03-01 | ... | True
60 | 2016-03-01 | ... | True

### Early Warning Style (missing means False)

If `include_missing_labels_in_train_as` is set to False, Triage treats the absence of a label row as a False label. The final matrix will look like:

entity_id | as_of_date | ...features... | label
------------ | ------------- | ------------- | ------------- 
25 | 2016-01-01 | ... | True
44 | 2016-01-01 | ... | **False**
25 | 2016-02-01 | ... | False
44 | 2016-02-01 | ... | True
25 | 2016-03-01 | ... | False
44 | 2016-03-01 | ... | True
60 | 2016-03-01 | ... | True

### Dropout Style (missing means True)

If `include_missing_labels_in_train_as` is set to True, Triage treats the absence of a label row as a True label. The final matrix will look like:

entity_id | as_of_date | ...features... | label
------------ | ------------- | ------------- | ------------- 
25 | 2016-01-01 | ... | True
44 | 2016-01-01 | ... | **True**
25 | 2016-02-01 | ... | False
44 | 2016-02-01 | ... | True
25 | 2016-03-01 | ... | False
44 | 2016-03-01 | ... | True
60 | 2016-03-01 | ... | True


If you would like to test how your cohort and label combine to make matrices, you can tell Triage to generate matrices and then inspect the matrices. To do this, we assume that you have your cohort and label defined in an experiment config file, as well as temporal config. The last piece needed to make matrices is some kind of features. Of course, the features aren't our main focus here, so let's use a placeholder feature that should create very quickly.


```
config_version: 'v6'

temporal_config:
    feature_start_time: '2010-01-04'
    feature_end_time: '2018-03-01'
    label_start_time: '2015-02-01'
    label_end_time: '2018-03-01'

    model_update_frequency: '1y'
    training_label_timespans: ['1month']
    training_as_of_date_frequencies: '1month'

    test_durations: '1month'
    test_label_timespans: ['1month']
    test_as_of_date_frequencies: '1month'

    max_training_histories: '5y'

cohort_config:
  query: |
    select distinct(entity_id)
    from permits
    where
    tsrange(start_time, end_time, '[]') @> {as_of_date}
  name: 'permits_in_last_year'

label_config:
  query: |
    select
    entity_id,
    bool_or(result = 'fail')::integer as outcome
    from inspections
    where '{as_of_date}'::timestamp <= date
    and date < '{as_of_date}'::timestamp + interval '{label_timespan}'
    group by entity_id
  name: 'failed_inspection'

feature_aggregations:
    -
        prefix: 'test'
        from_obj: 'permits'
        knowledge_date_column: 'date'
        aggregates_imputation:
            all:
                type: 'zero_noflag'
        aggregates: [{quantity: '1', metrics: ['sum']}]
        intervals: ['3month']
        groups: ['entity_id']
```

The above feature aggregation should just create a feature with the value `1` for each entity, but what's important here is that it's a valid feature config that allows us to make complete matrices. To make matrices using all of this configuration, you can run:

```python
from triage.experiments import SingleThreadedExperiment
from triage import create_engine
import yaml

with open('<your_experiment_config.yaml>') as fd:
    experiment_config = yaml.full_load(fd)
experiment = SingleThreadedExperiment(
    experiment_config=experiment_config,
    db_engine=create_engine(...),
    project_path='./'
)
experiment.generate_matrices()
```

The matrix generation process will run all of the cohort/label/feature generation above, and then save matrices to your project_path's `matrices` directory. By default, these are CSVs and should have a few columns: 'entity_id', 'date', 'test_1_sum', and 'failed_inspection'. The 'entity_id' and 'date' columns represent the index of this matrix, and 'failed_inspection' is the label.  Each of these CSV files has a YAML file starting with the same hash representing metadata about that matrix. If you want to look for just the train matrices to inspect the results of the `include_missing_labels_in_train_as` flag, try this command (assuming you can use bash):

```bash
$ grep "matrix_type: train" *.yaml
3343ebf255af6dbb5204a60a4390c7e1.yaml:matrix_type: train
6ee3cd406f00f0f47999513ef5d49e3f.yaml:matrix_type: train
74e2a246e9f6360124b96bea3115e01f.yaml:matrix_type: train
a29c9579aa67e5a75b2f814d906e5867.yaml:matrix_type: train
a558fae39238d101a66f9d2602a409e6.yaml:matrix_type: train
f5bb7bd8f251a2978944ba2b82866153.yaml:matrix_type: train
```

You can then open up those files and ensure that the labels for each entity_id/date pair match what you expect.

## Wrapup

Cohorts and Labels require a lot of care to define correctly as they constitute a large part of the problem framing. Even if you leave all of your feature generation the same, you can completely change the problem you're modeling by changing the label and cohort. Testing your cohort and label config can give you confidence that you're framing the problem the way you expect.
