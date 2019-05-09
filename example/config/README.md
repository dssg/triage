# Configuration

Explain the parameters for running Triage experiments

Triage is a great tool to make our life easier by semi-automagting many different tasks when we are doing predictive anlaytics projects, so that the usres can focus more on the problem formulation and modelling than the implementation. The configuration helps users define the parameters in an experiment. To run a full Triage experiment, users are required to define `experiment.yaml`, `feature.yaml` and `audition.yaml`. `postmodeling_config.yaml` and `postmodeling_crosstabs.yaml` are
optional, only for users who want to use `triage.postmodeling` module after experiment. 

## Experiment Configuration
Also check out the the example file `experiment.yaml`.

### Config Version

- `config_version`: The experiment configuration changes from time to time, and we upgrade the `triage.experiments.CONFIG_VERSION` variable whenever drastic changes that break old configuration files are released. Be sure to assign the config version that matches the `triage.experiments.CONFIG_VERSION` in the triage release you are developing against!

### Experiment Metadata
- `model_comment` (optional): will end up in the model_comment column of the models table for each model created in this experiment.
- `random_seed`: will be set in Python at the beginning of the experiment and affect the generation of all model seeds.

### Time Splitting
The time window to look at, and how to divide the window into train/test splits

- `temporal_config`:
    - `feature_start_time`: earliest date included in features
    - `feature_end_time`: latest date included in features
    - `label_start_time`: earliest date for which labels are avialable
    - `label_end_time`: day AFTER last label date (all dates in any model are before this date)
    - `model_update_frequency`: how frequently to retrain models
    - `training_as_of_date_frequencies`: time between as of dates for same entity in train matrix
    - `test_as_of_date_frequencies`: time between as of dates for same entity in test matrix
    - `max_training_histories`: length of time included in a train matrix
    - `test_durations`: length of time included in a test matrix (0 days will give a single prediction immediately after training end)
    - `training_label_timespans`: time period across which outcomes are labeled in train matrices
    - `test_label_timespans`: time period across which outcomes are labeled in test matrices

### Cohort Config
Cohorts are configured by passing a query with placeholders for the *as_of_date*.

- `cohort_conifg`: 
    - `qurey`: The `query` key should have a query, parameterized with an `'{as_of_date}'`, to select the entity_ids that should be included for a given date. The `{as_of_date}` will be replaced with each `as_of_date` that the experiment needs. The returned `entity_id` must be an integer.
    - `name`: You may enter a `name` for your configuration. This will be included in the metadata for each matrix and used to group models. If you don't pass one, the string `default` will be used.
