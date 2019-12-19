# Using `triage` for a Project: Workflow Tips

!!! warning "Getting Started..."
    The setup and first iteration here closely follow the [QuickStart Guide](quickstart.md), so that may be a good place to start if you're new to `triage`. 

    If you've already completed the QuickStart and have a working environment, you may want to jump ahead to [Iteration 2](#iteration-2-refine-the-cohort-and-temporal-setup)

## Step 1: Get your data set up
Triage needs data in a `Postgresql` database, with at least one table that contains `events` (one per row) and
`entities` of interest (people, place, organization, etc.; identified by an integer-valued `entity_id`), a `timestamp` (specifyinfg when the event occurred), and
additional attributes of interest about the event and/or entity (demographics for example).

We will need a database credentials either in a
[config file](https://github.com/dssg/triage/blob/master/example/database.yaml)
or as an environment variable called `DATABASE_URL`  that contains the
name  of the database, server, username, and password to use to
connect to it.

## Iteration 1: Quick Check

This set up will run a quick sanity check to make sure everything is set up correctly and that triage runs with your data and set up.

### Configuration
The full `triage` configuration file has a lot of sections allowing for extensive customization. In the first iteration, we'll set up the *minimal* parameters necessary to get started and make sure you have a working `triage` setup, but will describe how to better customize this configuration to your project in subsequent iterations. The starting point here is the [QuickStart `triage` config found here](quickstart.md#3-set-up-triage-configuration-files).

1. Define `outcome/label` of interest: This is a `SQL` query that
defines outcome we want to predict and needs to return The query must
return two columns: `entity_id` and `outcome`, based on a given
`as_of_date` and `label_timespan`. For more detail, see our [guide to
Labels](https://dssg.github.io/triage/experiments/cohort-labels/).
2. Define `label_timespan` ( = '12month' for example for predicting one year out)
3. Specify `features_aggregations` (at least one is neded to run `triage` -- for the QuickStart config, you need only specify an events table and we'll start with a simple count)
4. Specify `scoring` (i.e. evaluation metrics) to calculate (at least one is needed)
5. Specify `model_grid_preset` (i.e. model grid) for `triage` to run models with different hyperparameters (set to `'quickstart'` at this time)

### Run `triage`

1. Check triage config
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run the pipeline
```
triage experiment config.yaml --project-path '/project_directory'
```

Alternatively, you can also import `triage` as a package in your python scrips and run it that way. Learn more about that option [here](https://github.com/dssg/triage#using-triage).

### Check for Results

For this quick check, we're only running a handful of models for a single time window, so `triage`'s tools for model selection and postmodeling analysis won't be instructive, but you can confirm that by checking out the `triage`-created tables in the `model_metadata`, `test_results`, and `train-results` schemas in your database. In particular, you should find records for all your expected models in `model_metadata.models`, predictions from every model for every entity in `test_results.predictions`, and aggregate performance metrics in `test_results.evaluations` for every model.

If that all looks good, it's time to get started with customizing your modeling configuration to your project...


## Iteration 2: Refine the cohort and temporal setup

In this next iteration, we'll stick with a simplified feature configuration, but start to refine the parameters that control your modeling universe and cross-validation. We'll also introduce `audition`, `triage`'s component for model selection. However, with the limited feature set, we'll stick with the `quickstart` model grid for the time being.

### Define your cohort

For `triage`, the `cohort` represents the universe of relevant entities for a model at a given point in time. In the first iteration, we omitted the `cohort_config` section of our experiment configuration, in which case `triage` simply includes every entity it can find in your feature configuration. However, in most cases, you'll likely want to focus on a smaller set of entities, for instance:

- In a housing inspection project, you might want to include only houses that were built before a given modeling date and occupied at the modeling date

- In a project to help allocate housing subsidies, only certain individuals in your data might be eligible for the intervention at a given point in time

- In a recidivism prediction project, you might want to exclude individuals who are incarcerated as of the modeling date

You can specify a cohort in your config file with a `SQL` query that returns a set of integer `entity_id`s for a given modeling date (parameterized as `{as_of_date}` in your query). For instance:

```
cohort_config:
    query: "select entity_id from events where outcome_date < '{as_of_date}'"
    name: 'past_events'
```

### Configure your temporal settings

In most real-world machine learning applications, you're interested in training on past data and predicting forward into the future. `triage` is built with this common use-case in mind, relying on temporal cross-validation to evaluate your models' performance in a manner that best reflects how it will be used in practice.

There is a lot of nuance to the temporal configuration and it can take a bit of effort to get right. If you're new to `triage` (or want a refresher), we highly reccomend you check out [the temporal crossvalidation deep dive](dirtyduck/triage_intro.md#temporal-crossvalidation).

In previous iteration, we used a highly simplified temporal config, with just one parameter: `label_timespans`, yielding a single time split to get us started. However, these default values are generally not particularly meaningful in most cases and you'll need to fill out a more detailed `temporal_config`. Here's what that might look like: 

```
temporal_config:
    feature_start_time: '1995-01-01' # earliest date included in features
    feature_end_time: '2015-01-01'   # latest date included in features
    label_start_time: '2012-01-01' # earliest date for which labels are avialable
    label_end_time: '2015-01-01' # day AFTER last label date (all dates in any model are < this date)
    model_update_frequency: '6month' # how frequently to retrain models
    training_as_of_date_frequencies: '1day' # time between as of dates for same entity in train matrix
    test_as_of_date_frequencies: '3month' # time between as of dates for same entity in test matrix
    max_training_histories: ['6month', '3month'] # length of time included in a train matrix
    test_durations: ['0day', '1month', '2month'] # length of time included in a test matrix (0 days will give a single prediction immediately after training end)
    training_label_timespans: ['1month'] # time period across which outcomes are labeled in train matrices
    test_label_timespans: ['7day'] # time period across which outcomes are labeled in test matrices
```

For more detailed guidance on how to think about each of these parameters and set them for your context, [see the deep dive](dirtyduck/triage_intro.md#temporal-crossvalidation), but here are a couple of quick notes:

- The `feature_start_time` should reflect the earliest time available for your features, and will often be considerably earlier than the `label_start_time`.

- All of your train/test splits will be between the `label_start_time` and `label_end_time`, with splits starting from the last date and working backwards. Note that the `label_end_time` should be **1 day AFTER the last label date**.

- If you're using the same label timespan in both training and testing, you can still use the single `label_timespans` parameter (as we did in the QuickStart config). If you need different values, you can separately configure `test_label_timespans` and `training_label_timespans` (but note in this case, you should omit `label_timespans`).

- The parameters with plural names (e.g., `test_durations`) can be given as lists, in which case, `triage` will run models using all possible combinations of these values. This can get complicated fast, so you're generally best off starting with a single value for each parameter, for instance:

```
temporal_config:
    feature_start_time: '1980-01-01'
    feature_end_time: '2019-05-01'
    label_start_time: '2012-01-01'
    label_end_time: '2019-05-01'
    model_update_frequency: '1month'
    label_timespans: ['1y']
    max_training_histories: ['0d']
    training_as_of_date_frequencies: ['1y']
    test_as_of_date_frequencies: ['1y']
    test_durations: ['0d']
```

As you figure out your temporal parameters, you can use the `triage` CLI's `--show-timechop` parameter to visualize the resulting time splits:

```
triage experiment config.yaml --project-path '/project_directory' --show-timechop
```

### Run `triage`

1. Check triage config
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run the pipeline
```
triage experiment config.yaml --project-path '/project_directory'
```

Alternatively, you can also import `triage` as a package in your python scrips and run it that way. Learn more about that option [here](https://github.com/dssg/triage#using-triage).

### Check for Results

#### Check the database

As above, you should check the `triage`-created tables in your database to ensure the run with your new config has trained and tested all of the expected models. A couple of things to look out for:

- In `triage`, a specification of a model algorithm, related hyperparameters, and set of features is referred to as a `model_group` while an instantiation of these parameters on particular set of data at a specific point in time is referred to as a `model`. As such, with the `quickstart` preset model grid, you should still have the same 3 records in `model_metadata.model_groups` while you should have several new records in `model_metadata.models` with different `train_end_time`s implied by your temporal config.

- Likewise, in `test_results.predictions` and `test_results.evaluations`, you will find an `as_of_date` column. In many cases, you will likely have a single `as_of_date` per model that lines up with the model's `train_end_time`, but in some situations, you may want to evaluate at several `as_of_dates` for each model. [See the temporal crossvalidation deep dive](dirtyduck/triage_intro.md#temporal-crossvalidation) for more details.

#### A first look at model selection with `audition`

Now that we have models trained across several time periods, we can use `audition` to take a look at each `model_group`'s performance over time. While the `quickstart` models are quite simple and there isn't much meaningful model selection to do at this point, we can start to explore how model selection works in `triage`. A good place to begin is with the [model selection primer](dirtyduck/audition/).

We generally recommend using `audition` interactively with as a `jupyter notebook`. If you don't already have `jupyter` installed, you can learn more about it [here](https://jupyter.org/index.html). Once you have a notebook server running, you can modify the [`audition` tutorial notebook](https://github.com/dssg/triage/blob/master/src/triage/component/audition/Audition_Tutorial.ipynb) to take a look at the data from your current experiment. The [`audition README`](https://github.com/dssg/triage/tree/master/src/triage/component/audition) is also a good resource for options available with the tool.


## Iteration 3: Add more data/features, models and hyperparameters, and evaluation metrics of interest

After completing iteration 2, you should now have your cohort, label, and temporal configuration well-defined for your problem and you're ready to focus on features and model specifications. 

We've labeled this section `Iteration 3`, but in practice it's probably more like `Iterations 3-n` as you will likely want to do a bit of intermediate testing while adding new features and refine your model grid as you learn more about what does and doesn't seem to work well.

### Define some additional features

Generally speaking, the biggest determinant of the performance of many models is the quality of the underlying features, so you'll likely spend a considerable amount of time at this stage of the process. Here, you'll likely want to add additional features based on the data you've already prepared, but likely will discover that you want to structure or collect additional raw data as well where possible. 

The experiment configuration file provides a decent amount of flexibility for defining features, so we'll walk through some of the details here, however you may also want to refer to the relevant sections of the [config README](https://github.com/dssg/triage/blob/master/example/config/README.md#feature-generation) and [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) for more details.

!!! info "Features in `triage` are temporal aggregations"

    Just as `triage` is built with temporal cross-validation in mind, features in `triage` reflect this inherent temporal nature as well. As such, all feature definitions need to be specified with an associated date reflecting when the information was known (which may or may not be the same as when an event actually happened) and a time frame before the modeling date over which to aggregate.

    This has two consequences which may feel unintuitive at first:
    - Even static features are handled in this way, so in practice, we tend to specify them as a `max` (or `min`) taken over identical values over all time.

    - Categorical features are also aggregated over time in this way, so in practice these are split into separate features for each value the categorical can take, each of which is expressed as a numerical value (either binary or real-valued, like a mean over time). As a result, these values will not necessarily be mutually exclusive --- that is, a given entity can have non-zero values for more than one feature derived from the same underlying categorical depending on their history of values for that feature.

    - Unfortunately, `triage` has not yet implemented functionality for "first value" or "most recent value" feature aggregates, so you'll need to pre-calculate any features you want with this logic (though we do hope to add this ability).

Feature definitions are specified in the `feature_aggregations` section of the config file, under which you should provide a list of sets of related features, and each element in this list must contain several keys (see the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) for a detailed example of what this looks like in practice):
- `prefix`: a simple name to identify this set of related features - all of the features defined by this `feature_aggregations` entry will start with this prefix.
- `from_obj`: this may be a table name or a SQL expression that provides the source data for these features.
- `knowledge_date_column`: the date column specifying when information in the source data was known (e.g., available to be used for predictive modeling), which may differ from when the event ocurred.
- `aggregates` and/or `categoricals`: lists used to define the specific features (you must specify at least one, but may include both). See below for more detail on each.
- `intervals`: The time intervals (as a SQL interval, such a `'5 year'`, `'6 month'`, or `all` for all time) over which to aggregate features.
    - For instance, if you specified a count of the number of events under `aggregates` and `['5 year', '10 year', 'all']` as `intervals`, `triage` would create features for the number of events related to an entity in the last 5 years, 10 years, and since the `feature_start_time` (that is, three separate features)
- `groups`: levels at which to aggregate the features, often simply `entity_id`, but can also be used for other levels of analysis, such as spatial aggregations by zip codes, etc.
- You also need to provide rules for how to handle missing data, which can be provided either overall under `feature_aggregations` to apply to all features or on a feature-by-feature basis. It's worth reading through the [Feature Generation README](https://github.com/dssg/triage/blob/master/example/config/README.md#feature-generation) to learn about the available options here, including options for when missingness is meaningful (e.g., in a count) or there should be no missing data.

When defining features derived from numerical data, you list them under the `aggregates` key in your feature config, and these should include keys for:
- `quantity`: A column or SQL expression from the `from_obj` yielding a number that can be aggregated
- `metrics`: What types of aggregation to do. Namely, these are [postgres aggregation functions](https://www.postgresql.org/docs/9.5/functions-aggregate.html), such as `count`, `avg`, `sum`, `stddev`, `min`, `max`, etc.
- (optional) `coltype`: can be used to control the type of column used in the generated features table, but generally is not necessary to define.
- As noted above, imputation rules can be specified at this level as well.

When defining features derived from categorical data, you list them under the `categoricals` key in your feature config, and these should include keys for:
- `column`: The column containing the categorical information in the `from_obj` (note that this must be a column, not a SQL expression). May be any type of data, but the choice values specified must be compatible for equality comparisson in SQL.
- `choices` or `choice_query`: Either a hand-coded list of `choices` (that is, categorical values) or a `choice_query` that returns these distinct values from the data.
    - For categoricals with a very large number of possible unique values, you may want to limit the set of choices to a set of most frequently observed values.
    - Values in the data but not in this set of choice values will simply yield `0`s for all of these choice-set values.
- `metrics`: As above, the [postgres aggregation functions](https://www.postgresql.org/docs/9.5/functions-aggregate.html) used to aggregate values across the time intervals for the feature.
    - If categorical values associated with an entity **do not** change over time, using `max` would give you a simple one-hot encoded categorical.
    - If they are changing over time, `max` would give you something similar to a one-hot encoding, but note that the values would no longer be mutually-exclusive.
- As noted above, imputation rules can be specified at this level as well.

Much more detail about defining your features can be found in the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) and associated [README](https://github.com/dssg/triage/blob/master/example/config/README.md#feature-generation).

### Expand, then refine, your model grid

As you develop more features, you'll want to build out your modeling grid as well. Above, we've used the very sparse `quickstart` grid preset, but `triage` offers additional `model_grid_preset` options of varying size:
- The `small` model grid includes a reasonably extensive set of logistic regression and decision tree classifiers as well as a single random forest specification. This grid can be a good option as you build and refine your features, but you'll likely want to try something more extensive once you have the rest of your config set.
- The `medium` model grid is a good starting point for general modeling, including fairly extensive logistic regressions, decision trees, and random forest grids as well as a few ada boost and extra trees specification.
- The `large` grid adds additional specifications for these modeling types, including some very large (10,000-estimator) random forest and extra trees classifiers, so can take a bit more time and computational resources to run.

These preset grids should really serve as a starting point, and as you learn what seems to be working well in your use-case, you'll likely want to explore other specifications, which you can do by specifying your own `grid_config` in the triage config, which looks like:

```
grid_config:
    'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression':
        C: [0.00001,0.0001, 0.001, 0.01, 0.1, 10]
        penalty: ['l2']

    'sklearn.tree.DecisionTreeClassifier':
        criterion: ['entropy']
        max_depth: [null,2,5,10,50,100]
        min_samples_split: [2, 10, 50]
        min_samples_leaf: [0.01,0.05,0.10]
```

Here, each top-level key is the modeling package (this needs to be a classification algorithm with a `scikit-learn`-style interface, but need not come from `scikit-learn` specifically), and the keys listed under it are hyperparameters of the algorithm with a list of values to test. `triage` will run the grid of all possible combinations of these hyperparameter values. Note that you can't specify both a `model_grid_preset` and `grid_config` at the same time.

Check out the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) for more details on specifying your grid.

### Specify evaluation metrics you care about

In the initial iterations, we simply used precision in the top 1% as the evaluation metric for our models, but this is likely not what you care about for you project! Under the `scoring` section of your config file, you should specify the metrics of interest:

```
scoring:
    testing_metric_groups:
        -
          metrics: [precision@, recall@]
          thresholds:
            percentiles: [1,5,10]
            top_n: [100, 250, 500]
        -
          metrics: [accuracy, roc_auc]


    training_metric_groups:
      -
          metrics: [fpr@]
          thresholds:
            top_n: [100, 250, 500]
```

You can specify any number of evaluation metrics to be calculated for your models on either the training or test sets (the set of available metrics can be found [here](https://github.com/dssg/triage/blob/master/src/triage/component/catwalk/evaluation.py#L161)). For metrics that need to be calculated relative to a specific threshold in the score (e.g. precision), you must specify either `percentiles` or `top_n` (and can optionally provide both) at which to do the calculations.

Additionally, you can have `triage` pre-calculate statistics about bias and disparities in your modeling results by specifying a `bias_audit_config` section, which should give details about the attributes of interest (e.g., race, age, sex) and thresholds at which to do the calculations. See the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) and associated [README](https://github.com/dssg/triage/blob/master/example/config/README.md#bias-audit-config-optional) for more details on setting it up.

### Run `triage`

1. Check triage config
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run the pipeline
```
triage experiment config.yaml --project-path '/project_directory'
```

Alternatively, you can also import `triage` as a package in your python scrips and run it that way. Learn more about that option [here](https://github.com/dssg/triage#using-triage).

### Check for Results

1. Check the database (including aequitas tables)
2. Run `audition`
3. A first look at `postmodeling`


## Iteration 4: Explore additional labels/outcomes, feature group strategies, and calculation evaluation metrics on subsets of entities that may be of special interest

### Additional labels

### Feature group strategies

### Subsets

### Run `triage`

1. Check triage config
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run the pipeline
```
triage experiment config.yaml --project-path '/project_directory'
```

Alternatively, you can also import `triage` as a package in your python scrips and run it that way. Learn more about that option [here](https://github.com/dssg/triage#using-triage).

### Check for Results

1. Check the database
2. Run `audition`
3. Run `postmodeling`

