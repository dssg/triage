# Configuration

Explain the parameters for running Triage experiments

Triage is a great tool to make our life easier by semi-automating many different tasks when we are doing predictive anlaytics projects, so that the usres can focus more on the problem formulation and modeling than implementation. The configuration helps users define the parameters in an experiment. To run a full Triage experiment, users are required to define `experiment.yaml`, `feature.yaml` and `audition.yaml`. The `postmodeling_config.yaml` and `postmodeling_crosstabs.yaml` are
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
Cohorts are configured by passing a query with placeholders for the **as_of_date**.

- `cohort_conifg`: 
    - `qurey`: The `query` key should have a query, parameterized with an `'{as_of_date}'`, to select the entity_ids that should be included for a given date. The `{as_of_date}` will be replaced with each `as_of_date` that the experiment needs. The returned `entity_id` must be an integer.
    - `name`: You may enter a `name` for your configuration. This will be included in the metadata for each matrix and used to group models. If you don't pass one, the string `default` will be used.


### Label Generation
Labels are configured by passing a query with placeholders for the `as_of_date` and `label_timespan`.

- `label_config`:
    - `query`: The query must return two columns: `entity_id` and `outcome`, based on a given `as_of_date` and `label_timespan`. The `as_of_date` and `label_timespan` must be represented by placeholders marked by curly brackets. The example below reproduces the inspection outcome boolean-or logic: In addition, you can configure what label is given to entities that are in the matrix (see **cohort_config** section) but that do not show up in this label query. By default, these will show up as missing/null.
    - `include_missing_labels_in_train_as`: However, passing the key `include_missing_labels_in_train_as` allows you to pick True or False.
    - `name`: In addition to these configuration options, you can pass a name to apply to the label configuration that will be present in matrix metadata for each matrix created by this experiment, under the `label_name` key. The default label_name is `outcome`.


### Feature Generation
The aggregate features to generate for each train/test split. Implemented by wrapping [collate](https://github.com/dssg/collate). Most terminology here is taken directly from collate.

Each entry describes a collate.SpacetimeAggregation object, and the arguments needed to create it. Generally, each of these entries controls the features from one source table, though in the case of multiple groups may result in multiple output tables.

Rules specifying how to handle imputation of null values must be explicitly defined in your config file. These can be specified in two places: either within each feature or overall for each type of feature (aggregates_imputation, categoricals_imputation, array_categoricals_imputation). In either case, a rule must be given for each aggregation function (e.g., sum, max, avg, etc) used, or a catch-all can be specified with `all`. Aggregation function-specific rules will take precedence over the `all` rule and feature-specific rules will take precedence over the higher-level rules. Several examples are provided below.

Available Imputation Rules: 
    - `mean`: The average value of the feature (for SpacetimeAggregation the mean is taken within-date).
    - `constant`: Fill with a constant value from a required `value` parameter.
    - `zero`: Fill with zero.
    - `null_category`: Only available for categorical features. Just flag null values with the null category column.
    - `binary_mode`: Only available for aggregate column types. Takes the modal value for a binary feature.
    - `error`: Raise an exception if any null values are encountered for this feature.

- `feature_aggregations`: 
    - `prefix`: prefix given to the resultant tables
    - `from_obj`: from_obj is usually a source table but can be an expression, such as a join (ie 'cool_stuff join other_stuff using (stuff_id)')
    - `knowledge_date_column`: The date column to use for specifying which records to include in temporal features. It is important that the column used specifies the date at which the event is known about, which may be different from the date the event happened.
    - `aggregates_imputation`: top-level imputation rules that will apply to all aggregates functions can also specify `categoricals_imputation` or `array_categoricals_imputation`. You must specified at least one of the top-level or feature-level imputation to cover ever feature being defined.
        - `all`: The `all` rule will apply to all aggregation functions, unless overridden by more specific one
            - `type`: every imputation rule must have a `type` parameter, while some (like 'constant') have other required parameters (`value` here)
            - `value`
        - `max`: specifying `max` here will take precedence over the `all` rule for aggregations using a MAX() function
            - `type`:  
