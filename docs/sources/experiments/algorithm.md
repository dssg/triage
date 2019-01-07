# Experiment Algorithm Deep Dive

This guide's purpose is to provide familiarity of the inner workings of a Triage Experiment to people with some
experience in data science and Python. A Triage Experiment is a highly structured way of defining the experimentation
phase of a data science project. To those wondering whether this Experiment structure is flexible enough to fit their
needs, this should help.

## 1. Temporal Validation Setup

First, the given `temporal_config` section in the experiment definition is transformed into train and test splits,
including `as_of_times` for each matrix.

We create these splits by figuring out the latest reasonable split time from the inputs, and moving backwards in time
at the rate of the given `model_update_frequency`, until we get to the earliest reasonable split time.

For each split, we create `as_of_times` by moving either backwards from the split time towards the
`max_training_history` (for train matrices) or forwards from the split time towards the `test_duration` (for test
matrices) at the provided `data_frequency`.

Many of these configured values may be lists, in which case we generate the cross-product of all the possible values
and generate more splits.

For a more detailed look at the temporal validation logic, see [Temporal Validation Deep Dive](temporal-validation.md).

The train and test splits themselves are not used until the [Building Matrices](#3-building-matrices) section, but a
flat list of all computed `as_of_times` for all matrices needed in the experiment is used in the next section,
[Transforming Data](#2-transforming-data).

## 2. Transforming Data

With all of the `as_of_times` for this Experiment now computed, it's now possible to transform the input data into
features and labels as of all the required times.

### Labels

The Experiment populates a 'labels' table using the following input:
1. A query, provided by the user in the configuration file, that generates entity_ids and outcomes for a given
as_of_date and label_timespan.

2. Each as_of_date and label_timespan defined in temporal config

For instance, an inspections-style query (for the given timespan, return the entity and outcome of any matching
inspections) would look like:

```
	select
	events.entity_id,
	bool_or(outcome::bool)::integer as outcome
	from events
	where '{as_of_date}' <= outcome_date
		and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
		group by entity_id
```

This binary labels table is scoped to the entire Experiment, so all `as_of_time` (computed in step 1) and
`label_timespan` (taken straight from `temporal_config`) combinations are present. Additionally, the 'label_name' and
'label_type' are also recorded with each row in the table.

At this point, the 'labels' table may not have entries for all entities and dates that need to be in a given matrix.
How these rows have their labels represented is up to the configured `include_missing_labels_in_train_as` value in the
experiment. This value is not processed when we generate the labels table, but later on when the matrix is built (see
'Retrieving Data and Saving Completed Matrix')


### State Table

The Experiment keeps track of the state of the entities. Based on the configuration, certain entities can be included
or excluded for different time periods in feature imputation, creating matrices, or both.

In code, it does this by computing what it calls the 'sparse' state table for an experiment. This is a table with a
boolean flag entry for every entity, as_of_time, and state. The structure of this table allows for state filtering
based on SQL conditions given by the user.

Based on configuration, it can be created through one of three code paths:

1. If the user passes what we call a 'dense states' table, with the following structure: entity id/state/start/end, and
a list of state filters, this 'dense states' table holds time ranges that entities were in for specific states.
When converting this to a sparse table, we take each as_of_time present in the Experiment, and for each known state
(that is, the distinct values found in the 'dense states' table), see if there is any entry in the dense states table
with this state whose range overlaps with the as_of_time. If so, the entity is considered to be in that state as of that
date.

2. If the user passes what we call an 'entities' table, containing an entity_id, it will simply use all distinct
entities present in said table, and mark them as 'active' for every as_of_time in the experiment. Any other columns are
ignored.

3. If the user passes a query, parameterized with an as of date, we will populate the table by running it for each
as_of_date.

This table is created and exists until matrices are built, at which point it is considered unnecessary and then dropped.


### Features

Each provided `feature_aggregation` configures the creation and population of several feature tables in the 'features'
schema: one for each of the groups specified in the config, one that merges the groups together into one table, and one
that fills in null values from the merged table with imputed values based on imputation config.

#### Generating Aggregation SQL
To generate the SQL that creates the pre-imputation table, the Experiment assembles building blocks from the feature
aggregation config, as well as the experiment's list of `as_of_times`:

* `from_obj` represents, well, the object of the FROM clause in the SQL query. Often this is just a table, but can be
configured to be a subquery. This holds all the data that we want to aggregate into features
* Each `as_of_time` in the experiment and `interval` in the `feature_aggregation` is combined with the
`knowledge_date_column` to create a WHERE clause representing a valid window of events to aggregate in the `from_obj`:
e.g (`where {knowledge_date_column} >= {as_of_time} - interval {interval}`)
* Each `aggregate`, `categorical`, or `array_categorical` represents a SELECT clause. For aggregates, the `quantity` is
a column or SQL expression representing a numeric quantity present in the `from_obj`, and the `metrics` are any number
of aggregate functions we want to use. The aggregate function is applied to the quantity.
* Each `group` is a column applied to the GROUP BY clause. Generally this is 'entity_id', but higher-level groupings
(for instance, 'zip_code') can be used as long as they can be rolled up to 'entity_id'.
* By default the query is joined with the cohort table (see 'state table' above) to remove unnecessary rows. If `save_all_features` is passed to the Experiment this is not done.

So a simplified version of a typical query would look like:
```
SELECT {group}, {metric}({quantity})
FROM {from_obj}
JOIN {cohort_table} ON (
    {cohort_table.entity_id} = {from_obj.entity_id}
    AND {cohort_table.date} = {as_of_time}
)
WHERE {knowledge_date_column} >= {as_of_time} - interval {interval}
GROUP BY {group}
```

#### Writing Group-wide Feature Tables
For each `as_of_time`, the results from the generated query are written to a table whose name is prefixed with the
`prefix`, and suffixed with the `group`. For instance, if the configuration specifies zipcode-level aggregates and
entity-level aggregates, there will be a table for each, keyed on its group plus the as_of_date.

#### Merging into Aggregation-wide Feature Tables
Each generated group table is combined into one representing the whole aggregation with a left join. Given that the
groups originally came from the same table (the `from_obj` of the aggregation) and therefore we know the zipcode for
each entity, what we do now is create a table that would be keyed on entity and as_of_date, and contain all
entity-level and zipcode-level aggregates from both tables. This aggregation-level table represents all of the features
in the aggregation, pre-imputation. Its output location is generally `{prefix}_aggregation`

#### Imputing Values
A table that looks similar, but with imputed values is created. The state table from above is passed into collate as
the comprehensive set of entities and dates for which output should be generated, regardless if they exist in the
`from_obj`. Each feature column has an imputation rule, inherited from some level of the feature definition. The
imputation rules that are based on data (e.g. `mean`) use the rows from the `as_of_time` to produce the imputed value.
Its output location is generally `{prefix}_aggregation_imputed`

### Recap

At this point, we have at least three tables that are used to populate matrices:

- `labels` with computed labels
- `tmp_states_{experiment hash}` that tracks what `as_of_times` each entity was in each state.
- A `features.{prefix}_aggregation_imputed` table for each feature aggregation present in the experiment config.


## 3. Building Matrices

At this point, we have to build actual train and test matrices that can be processed by machine learning algorithms, save at the user's specified path, either on the local filesystem or s3 depending on the scheme portion of the path (e.g. `s3://bucket-name/project_directory`)

First we have to figure out exactly what matrices we have to build. The split definitions from step 1 are a
good start -- they are our train and test splits -- but sometimes we also want to test different subsets of the data,
like feature groups (e.g. 'how does using group of features A perform against using all features?'). So there's a layer
of iteration we introduce for each split, that may produce many more matrices.

What do we iterate over?
* Feature List - All subsets of features that the user wants to cycle through. This is the end result of the feature
group generation and mixing process, which is described more below.
* Cohorts - In theory we can take in different cohorts and iterate in the same experiment.  This is not fully implemented, so in reality we just use the one cohort that is passed in the `cohort_config`
* Label names - In theory we can take in different labels (e.g. complaints, sustained complaints) in the same
experiment. Right now there is no support for multiple label names, but the label name used is configurable through the
optional 'label_config'->'name' config value
* Label types - In theory we can take in different label types (e.g. binary) in the same experiment. *Right now this
isn't done, there is one label type and it is hardcoded as 'binary'.*

### Feature Lists
How do we arrive at the feature lists? There are two pieces of config that are used: `feature group_definition` and
`feature_group_strategies`. Feature group definitions are just ways to define logical blocks of features, most often
features that come from the same source, or describing a particular type of event. These groups within the experiment
as a list of feature names, representing some subset of all potential features for the experiment. Feature group
strategies are ways to take feature groups and mix them together in various ways. The feature group strategies take
these subsets of features and convert them into another list of subsets of features, which is the final list iterated
over to create different matrices.

#### Feature Group Definition
Feature groups, at present, can be defined as either a `prefix` (the prefix of the feature name), a `table` (the
feature table that the feature resides in), or `all` (all features).  Each argument is passed as a list, and each entry
in the list is interpreted as a group. So, a feature group config of `{'table': ['complaints_aggregate_imputed',
'incidents_aggregate_imputed']}` would result in two feature groups: one with all the features in
`complaints_aggregate_imputed`, and one with all the features in `incidents_aggregate_imputed`. Note that this requires
a bit of knowledge on the user's part of how the feature table names will be constructed.

`prefix` works on the prefix of the feature name as it exists in the database. So this also requires some knowledge of
how these get created. The general format is: `{aggregation_prefix}_{group}_{timeperiod}_{quantity}`, so with some
knowledge the user can create groups with the aggregation's configured prefix (common), or the aggregations configured
prefix + group (in case they want to compare, for instance, zip-code level features versus entity level features).

`all`, with a single value of `True`, will include a feature group with all defined features. If no feature group
definition is sent, this is the default.

Either way, at the end of this process the experiment will be aware of some list of feature groups, even if the list is
just length 1 with all features as one group.

#### Feature Group Mixing
A few basic feature group mixing strategies are implemented: `leave-one-in`, `leave-one-out`, and `all`. These are sent
in the experiment definition as a list, so different strategies can be tried in the same experiment. Each included
strategy will be applied to the list of feature groups from the previous step, to convert them into

For instance, 'leave-one-in' will cycle through each feature group, and for each one create a list of features that
just represents that feature group, so for some matrices we would only use features from that particular group.
`leave-one-out` does the opposite, for each feature group creating a list of features that includes all other feature
groups but that one. `all` just creates a list of features that represents all feature groups together.

### Iteration and Matrix Creation

At this point, matrices are created by looping through all train/test splits and data subsets (e.g. feature groups,
state definitions), grabbing the data corresponding to each from the database, and assembling that data into a design
matrix that is saved along with the metadata that defines it.

As an example, if the experiment defines 3 train/test splits (one test per train in this example, for simplicity), 3
feature groups that are mixed using the 'leave-one-out' and 'all' strategies, and 1 state definition, we'll expect 18
matrices to be saved: 9 splits after multiplying the time splits by the feature groups, and each one creating a train
and test matrix.

### Associating Matrices with Experiment

After all matrices for the Experiment are defined but before any are built, the Experiment is associated with each Matrix in the database through the `model_metadata.experiment_matrices` table. This means that whether or not the Experiment has to end up building a matrix, after the fact a user can query the database to see if it used said matrix.

#### Retrieving Data and Saving Completed Matrix

Each matrix that has to be built (i.e. has not been built by some prior experiment) is built by retrieving its data out of the database.

How do we get the data for an individual matrix out of the database?

1. Create an entity-date table for this specific matrix. There is some logic applied to decide what rows show up. There
are two possible sets of rows that could show up.

- `all valid entity dates`. These dates come from the entity-date-state table for the experiment (populated using the
rules defined in the 'cohort_config'), filtered down to the entity-date pairs that match both *the state filter and the
list of as-of-dates for this matrix*.

- `all labeled entity dates`. These dates consist of all the valid entity dates from above, that also have an entry in
the labels table.

If the matrix is a test matrix, all valid entity dates will be present.

If the matrix is a train matrix, whether or not valid but unlabeled examples show up is decided by the
`include_missing_labels_in_train_as` configuration value. If it is present in any form, these labels will be in the
matrix. Otherwise, they will be filtered out.

2. Write features data from tables to disk in CSV format using a COPY command, table by table. Each table is joined
with the matrix-specific entity-date table to only include the desired rows.

3. Write labels data to disk in CSV format using a COPY command. These labels will consist of the rows in the
matrix-specific entity-date table left joined to the labels table. Rows not present in the labels table will have their
label filled in (either True or False) based on the value of the `include_missing_labels_in_train_as` configuration key.

4. Merge the features and labels CSV files horizontally, in pandas. They are expected to be of the same shape, which is
enforced by the entity-date table. The resulting matrix is indexed on `entity_id` and `as_of_date`, and then saved to
disk (in CSV format, more formats to come) along with its metadata: time, feature, label, index, and state information.
along with any user metadata the experiment config specified. The filename is decided by a hash of this metadata, and
the metadata is saved in a YAML file with the same hash and directory. The metadata is additionally added to a database table 'matrices'.

Matrix metadata reference:
- [Train matrix temporal info](https://github.com/dssg/timechop/blob/master/timechop/timechop.py#L433-L440)
- [Test matrix temporal info](https://github.com/dssg/timechop/blob/master/timechop/timechop.py#L514-L523)
- [Feature, label, index, cohort, user
metadata](https://github.com/dssg/triage/blob/master/src/triage/component/architect/planner.py#L89-L112)

### Recap

At this point, all finished matrices and metadata will be saved under the `project_path` supplied by the user to the
Experiment constructor, in the subdirectory `matrices`.

## 4. Running Models

The last phase of an Experiment run uses the completed design matrices to train, test, and evaluate classifiers. This procedure writes a lot of metadata to the 3 schemas: 'model_metadata', 'train_results', and 'test_results'.

### Associating Models with Experiment

Every combination of training matrix + classifier + hyperparameter is considered a Model. Before any Models are trained, the Experiment is associated with each Model in the database through the `model_metadata.experiment_models` table. This means that whether or not the Experiment has to end up training a model, after the fact a user can query the database to see if it used said model.

### Train

Each matrix marked for training is sent through the configured grid in the experiment's `grid_config`. This works much
like the scikit-learn `ParameterGrid` (and in fact uses it on the backend). It cycles through all of the classifiers
and hyperparameter combinations contained herein, and calls `.fit()` with that train matrix. Any classifier that
adheres to the scikit-learn `.fit/.transform` interface and is available in the Python environment will work here,
whether it is a standard scikit-learn classifier, a third-party library like XGBoost, or a custom-built one in the
calling repository (for instance, one that implements the problem domain's baseline heuristic algorithm for
comparison).  Metadata about the trained classifier is written to the `model_metadata.models` Postgres table. The trained model is saved to a filename with the model hash (see Model Hash section below).

#### Model Groups

Each model is assigned a 'model group'. A model group represents a number of trained classifiers that we want to treat
as equivalent by some criteria. By default, this is aimed at defining models which are equivalent across time splits,
to make analyzing model stability easier. This default is accomplished with a set of 'model group keys' that includes
data about the classifier (module, hyperparameters), temporal intervals used to create the train matrix (label
timespan, training history, as-of-date frequency), and metadata describing the data in the train matrix (features and
feature groups, label name, cohort name). The user can override this set of `model_group_keys` in the experiment
definition, with all of the default information plus other matrix metadata at their disposal (See end of 'Retrieving
Data and Saving Completed Matrix' section for more about matrix metadata). This data is stored in the `model_metadata.model_groups` table, along with a `model_group_id` that is used as a foreign key in the `model_metadata.models` table.

#### Model Hash
Each trained model is assigned a hash, for the purpose of uniquely defining and caching the model. This hash is based
on the training matrix metadata, classifier path, hyperparameters (except those which concern execution and do not
affect results of the classifier, such as `n_jobs`), and the given project path for the Experiment. This hash can be
found in each row of the `model_metadata.models` table. It is enforced as a unique key in the table.

#### Global Feature Importance
The training phase also writes global feature importances to the database, in the `train_results.feature_importances` table.
A few methods are queried to attempt to compute feature importances:
* The bulk of these are computed using the trained model's `.feature_importances_` attribute, if it exists.
* For sklearn's `SVC` models with a linear kernel, the model's `.coef_.squeeze()` is used.
* For sklearn's LogisticRegression models, `np.exp(model.coef_).squeeze()` is used.
* Otherwise, no feature importances are written.


### Test Matrix
For each test matrix, predictions, individual importances, and the user-specified testing evaluation metrics are written to the 'test_results' schema. For each train matrix, predictions and the user-specified training evaluation metrics are written to the 'train_results' schema.

#### Predictions
The trained model's prediction probabilities (`predict_proba()`) are computed both for the matrix it was trained on and any testing matrices. The predictions for the training matrix are saved in `train_results.predictions` and those for the testing matrices are saved in the `test_results.predictions`. More specifically, `predict_proba` returns the probabilities for each label (false and true), but in this case only the probabilities for the true label are saved in the `{train or test}_predictions` table. The `entity_id` and `as_of_date` are retrieved from the matrix's index, and stored in the database table along with the probability score, label value (if it has one), as well as other metadata.

### Individual Feature Importance
Feature importances (of a configurable number of top features, defaulting to 5) for each prediction are computed and written to the `test_results.individual_importances` table. Right now, there are no sophisticated calculation methods integrated into the experiment; simply the top 5 global feature importances for the model are copied to the `individual_importances` table.

#### Metrics
Triage allows for the computation of both testing set and training set evaluation metrics. Evaluation metrics, such as precision and recall at various thresholds, are written to either the `train_results.evaluations` table or the `test_results.evaluations`. Triage defines a number of [Evaluation Metrics](https://github.com/dssg/triage/blob/master/src/triage/component/catwalk/evaluation.py#L45-L58) metrics that can be addressed by name in the experiment definition, along with a list of thresholds and/or other parameters (such as the 'beta' value for fbeta) to iterate through. Thresholding is done either via absolute value (top k) or percentile by sorting the predictions and labels by the row's predicted probability score, with ties broken at random (the random seed can be passed in the config file to make this deterministic), and assigning the predicted value as True for those above the threshold. Note that the percentile thresholds are in terms of the population percentage, not a cutoff threshold for the predicted probability.

Sometimes test matrices may not have labels for every row, so it's worth mentioning here how that is handled and interacts with thresholding. Rows with missing labels are not considered in the metric calculations, and if some of these rows are in the top k of the test matrix, no more rows are taken from the rest of the list for consideration. So if the experiment is calculating precision at the top 100 rows, and 40 of the top 100 rows are missing a label, the precision will actually be calculated on the 60 of the top 100 rows that do have a label. To make the results of this more transparent for users, a few extra pieces of metadata are written to the evaluations table for each metric score.

* `num_labeled_examples` - The number of rows in the test matrix that have labels
* `num_labeled_above_threshold` - The number of rows above the configured threshold for this metric score that have
labels
* `num_positive_labels` - The number of positive labels in the test matrix

### Recap
At this point, the 'model_metadata', 'train_results', and 'test_results' database schemas are fully populated with data about models, model groups, predictions, feature importances, and evaluation metrics for the researcher to query. In addition, the trained model pickle files are saved in the configured project path. The experiment is considered finished.
