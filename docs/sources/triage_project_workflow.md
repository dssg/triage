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


(note imputation required -- if shouldn't have missing, use error; if missing means not present use zero no flag)

### Expand, then refine, your model grid

(call-out box for grid options)

(might start with `small`, but `medium` is a decent starting point for modeling in general. Once you have things well-refined, could consider `larger` if you have sufficient time and computational resources)

### Consider additional evaluation metrics

(also, add a aequitas config to obtain results about fairness)

### Run `triage`

1. Check triage config
```
triage experiment config.yaml --project-path '/project_directory' --validate-only
```

2. Run the pipeline
```
triage experiment config.yaml --project-path '/project_directory'
```

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

### Check for Results

1. Check the database
2. Run `audition`
3. Run `postmodeling`

