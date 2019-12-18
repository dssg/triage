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

(we're sticking with one feature for the moment, so no need to expand the model grid beyond `quickstart` here)

### Define your cohort

### Configure your temporal 

### Run `triage`

### Check for Results

1. Check the database
2. A first look at `audition`


## Iteration 3: Add more data/features, models and hyperparameters, and evaluation metrics of interest

(you'll probably want to add features do some testing and running as you add features, so in practice this is more like iterations 3--*n*...)

### Define some additional features

(may involve structuring or collecting more raw data, of course)

### Expand your model grid

(might start with `small`, but `medium` is a decent starting point for modeling in general. Once you have things well-refined, could consider `larger` if you have sufficient time and computational resources)

### Consider additional evaluation metrics

(also, add a aequitas config to obtain results about fairness)


## Iteration 4: Explore additional labels/outcomes and calculation evaluation metrics on subsets of entities that may be of special interest

### Additional labels

### Subsets
