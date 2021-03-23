# Using `triage` for a Project: Workflow Tips

!!! warning "Getting Started..."
    The setup and first iteration here closely follow the [QuickStart Guide](quickstart.md), so that may be a good place to start if you're new to `triage`.

    If you've already completed the QuickStart and have a working environment, you may want to jump ahead to [Iteration 2](#iteration-2-refine-the-cohort-and-temporal-setup)

## Step 1: Get your data set up
Triage needs data in a `Postgresql` database, with at least one table that contains `events` (one per row) and
`entities` of interest (people, place, organization, etc.; identified by an integer-valued `entity_id`), a `timestamp` (specifying when the event occurred), and
additional attributes of interest about the event and/or entity (demographics for example).

We will need database credentials either in a
[config file](https://github.com/dssg/triage/blob/master/example/config/database.yaml)
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

For this quick check, we're only running a handful of models for a single time window, so `triage`'s tools for model selection and postmodeling analysis won't be instructive, but you can confirm that by checking out the `triage`-created tables in the `triage_metadata`, `test_results`, and `train-results` schemas in your database. In particular, you should find records for all your expected models in `triage_metadata.models`, predictions from every model for every entity in `test_results.predictions`, and aggregate performance metrics in `test_results.evaluations` for every model.

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

### Set a `random_seed`

You may want to set an integer-valued `random_seed` for python to use in your configuration file in order to ensure reproducibility of your results across `triage` runs.

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

- In `triage`, a specification of a model algorithm, related hyperparameters, and set of features is referred to as a `model_group` while an instantiation of these parameters on particular set of data at a specific point in time is referred to as a `model`. As such, with the `quickstart` preset model grid, you should still have the same 3 records in `triage_metadata.model_groups` while you should have several new records in `triage_metadata.models` with different `train_end_time`s implied by your temporal config.

- Likewise, in `test_results.predictions` and `test_results.evaluations`, you will find an `as_of_date` column. In many cases, you will likely have a single `as_of_date` per model that lines up with the model's `train_end_time`, but in some situations, you may want to evaluate at several `as_of_dates` for each model. [See the temporal crossvalidation deep dive](dirtyduck/triage_intro.md#temporal-crossvalidation) for more details.

#### A first look at model selection with `audition`

Now that we have models trained across several time periods, we can use `audition` to take a look at each `model_group`'s performance over time. While the `quickstart` models are quite simple and there isn't much meaningful model selection to do at this point, we can start to explore how model selection works in `triage`. A good place to begin is with the [model selection primer](https://dssg.github.io/triage/audition/audition_intro/).

We generally recommend using `audition` interactively with as a `jupyter notebook`. If you don't already have `jupyter` installed, you can learn more about it [here](https://jupyter.org/index.html). Once you have a notebook server running, you can modify the [`audition` tutorial notebook](https://github.com/dssg/triage/blob/master/src/triage/component/audition/Audition_Tutorial.ipynb) to take a look at the data from your current experiment. The [`audition README`](https://github.com/dssg/triage/tree/master/src/triage/component/audition) is also a good resource for options available with the tool.


## Iteration 3: Add more data/features, models and hyperparameters, and evaluation metrics of interest

After completing iteration 2, you should now have your cohort, label, and temporal configuration well-defined for your problem and you're ready to focus on features and model specifications.

We've labeled this section `Iteration 3`, but in practice it's probably more like `Iterations 3-n` as you will likely want to do a bit of intermediate testing while adding new features and refine your model grid as you learn more about what does and doesn't seem to work well.

### Define some additional features

Generally speaking, the biggest determinant of the performance of many models is the quality of the underlying features, so you'll likely spend a considerable amount of time at this stage of the process. Here, you'll likely want to add additional features based on the data you've already prepared, but likely will discover that you want to structure or collect additional raw data as well where possible.

The experiment configuration file provides a decent amount of flexibility for defining features, so we'll walk through some of the details here, however you may also want to refer to the relevant sections of the [config README](http://dssg.github.io/triage/experiments/experiment-config#feature-generation) and [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) for more details.

!!! info "Features in `triage` are temporal aggregations"

    Just as `triage` is built with temporal cross-validation in mind, features in `triage` reflect this inherent temporal nature as well. As such, all feature definitions need to be specified with an associated date reflecting when the information was known (which may or may not be the same as when an event actually happened) and a time frame before the modeling date over which to aggregate.

    This has two consequences which may feel unintuitive at first:
    - Even static features are handled in this way, so in practice, we tend to specify them as a `max` (or `min`) taken over identical values over all time.

    - Categorical features are also aggregated over time in this way, so in practice these are split into separate features for each value the categorical can take, each of which is expressed as a numerical value (either binary or real-valued, like a mean over time). As a result, these values will not necessarily be mutually exclusive --- that is, a given entity can have non-zero values for more than one feature derived from the same underlying categorical depending on their history of values for that feature.

    - Unfortunately, `triage` has not yet implemented functionality for "first value" or "most recent value" feature aggregates, so you'll need to pre-calculate any features you want with this logic (though we do hope to add this ability).

Feature definitions are specified in the `feature_aggregations`
section of the config file, under which you should provide a list of
sets of related features, and each element in this list must contain
several keys (see the [example config
file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml)
for a detailed example of what this looks like in practice):

- `prefix`: a simple name to identify this set of related features - all of the features defined by this `feature_aggregations` entry will start with this prefix.
- `from_obj`: this may be a table name or a SQL expression that provides the source data for these features.
- `knowledge_date_column`: the date column specifying when information in the source data was known (e.g., available to be used for predictive modeling), which may differ from when the event ocurred.
- `aggregates` and/or `categoricals`: lists used to define the specific features (you must specify at least one, but may include both). See below for more detail on each.
- `intervals`: The time intervals (as a SQL interval, such a `'5 year'`, `'6 month'`, or `all` for all time) over which to aggregate features.
    - For instance, if you specified a count of the number of events under `aggregates` and `['5 year', '10 year', 'all']` as `intervals`, `triage` would create features for the number of events related to an entity in the last 5 years, 10 years, and since the `feature_start_time` (that is, three separate features)
- `groups`: levels at which to aggregate the features, often simply `entity_id`, but can also be used for other levels of analysis, such as spatial aggregations by zip codes, etc.
- You also need to provide rules for how to handle missing data, which can be provided either overall under `feature_aggregations` to apply to all features or on a feature-by-feature basis. It's worth reading through the [Feature Generation README](dssg.github.io/triage/experiments/experiment-config#feature-generation) to learn about the available options here, including options for when missingness is meaningful (e.g., in a count) or there should be no missing data.

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

Much more detail about defining your features can be found in the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) and associated [README](dssg.github.io/triage/experiments/experiment-config#feature-generation).

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

Additionally, you can have `triage` pre-calculate statistics about bias and disparities in your modeling results by specifying a `bias_audit_config` section, which should give details about the attributes of interest (e.g., race, age, sex) and thresholds at which to do the calculations. See the [example config file](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) and associated [README](http://dssg.github.io/triage/experiments/experiment-config#bias-audit-config-optional) for more details on setting it up.

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

### Check for Results and Select Models

#### Check the database

As before, the first place to look to check on the results of your modeling run is in the database:
- Even while the modeling is still running, you can check out `test_results.evaluations` to keep an eye on the progress of the run (join it to `triage_metadata.models` using `model_id` if you want to see information about the model specifications).
- Once the run has finished, you should see many more models in `test_results.evaluations` reflecting the full model grid evaluated on each of the metrics you specified above.
- Information on feature importances can be found in `train_results.feature_importances` (note the schema is `train_results` since these are calculated based on the training data).

#### Run `audition`

Once you have a more comprehensive model run with a variety of features and modeling grid, `audition` can help you understand the performance of different specifications and further refine your models for future iterations. In a typical project, you'll likely run through the `audition` flow several times as you progressively improve your modeling configuration.

When you finally settle on a configuration you're happy with, `audition` will also help you narrow your models down to a smaller set of well-performing options for futher analysis. Often, this might involve something like specifying a few different "selection rules" (e.g., best mean performance, recency-weighted performance, etc.) and exploring one or two of the best performing under each rule using `postmodeling`.

More about using `audition`:
- [model selection primer](dirtyduck/audition/).
- [`audition` tutorial notebook](https://github.com/dssg/triage/blob/master/src/triage/component/audition/Audition_Tutorial.ipynb)
- [`audition` README](https://github.com/dssg/triage/tree/master/src/triage/component/audition)

#### A first look at `postmodeling`

Now that you've narrowed your grid down to a handful of model specification for a closer look, the `postmodeling` methods provided in `triage` will help you answer three avenues of investigation:

- Dive deeper into what’s going on with each of these models, such as:
    - score and feature distributions
    - feature importances
    - performance characteristics, such as stack ranking, ROC curves, and precision-recall curves

- Debug and improve future models
    - look for potential leakage of future information into your training set
    - explore patterns in the model's errors
    - identify hyperparameter values and features to focus on in subsequent iterations

- Decide how to proceed with deployment
    - compare lists and important features across models
    - help decide on either a single "best" model to deploy or a strategy that combines models

Like `audition`, our `postmodeling` tools are currently best used interactively with a `jupyter notebook`. You can read more about these tools in the [`postmodeling` README](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/README.md) and modify the [example `postmodeling` notebook](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/postmodeling_tutorial.ipynb) for your project.


## Iteration 4: Explore additional labels/outcomes, feature group strategies, and calculation evaluation metrics on subsets of entities that may be of special interest

Finally, in `Iteration 4`, you should consider exploring additional labels, `triage`'s tools for understanding feature contributions, and potentially look at evaluating your models on subsets of interest in your cohort.

### Additional labels

In many projects, how you choose to define your outcome label can have a dramatic impact on which entities your models bring to the top, as well as disparities across protected groups. As such, we generally recommend exploring a number of options for your label definition in the course of a given project. For instance:

- In a project to target health and safety inspections of apartment buildings, you might consider labels that look at the presence of any violation, the presence of at least X violations, violations of a certain type or severity, violations in a certain fraction of units, etc.

- In a recidivism prediction project, you might consider labels that focus on subsequent arrests for any reason or only related to new criminal activity; based on either arrests, bookings, or convictions; or related to certain types or severity of offense.

- In a health care project, you might consider re-hospitalizations over differ time frames, certain types of complications or severity of outcomes or prognoses.

Be sure to change the `name` parameter in your `label_config` with each version to ensure that `triage` recognizes that models built with different labels are distinct.

### Feature group strategies

If you want to get a better sense for the most important types of features in your models, you can specify a `feature_group_strategies` key in your configuration file, allowing you to run models that include subsets of your features (note that these are taken over your feature groups --- often the different `prefix` values you specified --- not the individual features).

The strategies you can use are: `all`, `leave-one-out`, `leave-one-in`, `all-combinations`. You can specify a list of multiple strategies, for instance:

```
feature_group_strategies: ['all', 'leave-one-out']
```

If you had five feature groups, this would run a total of six strategies (one including all your feature groups, and five including all but one of them) for each specification in your model grid.

!!! warning "Before using feature group stragies..."
    Note that model runs specifying `feature_group_strategies` can become quite time and resource-intensive, especially using the `all-combinations` option.

    Before making use of this functionality, it's generally smart to narrow your modeling grid considerably to at most a handful of well-performing models and do some back-of-the-envelope calculations of how many variations `triage` will have to run.

Learn more about feature groups and strategies in the [config README](dssg.github.io/triage/experiments/experiment-config#feature-grouping-optional).

### Subsets

In some cases, you may be interested in your models' performance on subsets of the full cohort on which it is trained, such as certain demographics or individuals who meet a specific criteria of interest to your program (for instance, a certain level or history of need).

Subsets are defined in the `scoring` section of the configuration file as a list of dictionaries specifying a `name` and `query` that identify the set of entities for each subset of interest using `{as_of_date}` as a placeholder for the modeling date.

Here's a quick example:

```
scoring:

    ...

    subsets:
        -
            name: women
            query: |
                select distinct entity_id
                from demographics
                where d.gender = 'woman'
                and demographic_date < '{as_of_date}'::date
        -
            name: youts
            query: |
                select distinct entity_id
                from demographics
                where extract('year' from age({as_of_date}, d.birth_date)) <= 18
                and demographic_date < '{as_of_date}'::date
```

When specify subsets, all of the model evaluation metrics will be calculated for each subset you define here, as well as the cohort overall. In the `test_results.evaluations` table, the `subset_hash` column will identify the subset for the evaluation (`NULL` values indicate evaluations on the entire cohort), and can be joined to `triage_metadata.subsets` to obtain the name and definition of the subset.

Note that subsets are only used for the purposes of evaluation, while the model will still be trained and scored on the entire cohort described above.

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

### Check for Results and Select Models

As described above, once your modeling run has completed, you can explore the results in the `triage`-generated tables in your database, perform model selection with `audition`, and dig deeper into your results with `postmodeling`:

#### Check the database

Look for results and associated information in:
- `triage_metadata`
- `train_results`
- `test_results`

#### Run `audition`

More about using `audition`:

- [model selection primer](dirtyduck/audition/).
- [`audition` tutorial notebook](https://github.com/dssg/triage/blob/master/src/triage/component/audition/Audition_Tutorial.ipynb)
- [`audition` README](https://github.com/dssg/triage/tree/master/src/triage/component/audition)

#### Run `postmodeling`

More about `postmodeling`:

- [`postmodeling` README](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/README.md)
- [example `postmodeling` notebook](https://github.com/dssg/triage/blob/master/src/triage/component/postmodeling/contrast/postmodeling_tutorial.ipynb)
