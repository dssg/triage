# Configuration

Explain the parameters for running Triage experiments

Triage is a great tool to make our life easier by semi-automating many different tasks when we are doing predictive anlaytics projects, so that the usres can focus more on the problem formulation and modeling than implementation. The configuration helps users define the parameters in an experiment. To run a full Triage experiment, users are required to define `experiment.yaml` and `audition.yaml`. The `postmodeling_config.yaml` and `postmodeling_crosstabs.yaml` are
optional, only for users who want to use `triage.postmodeling` module after experiment. 

## Experiment Configuration
Also check out the example file `experiment.yaml`.

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
The aggregate features to generate for each train/test split. Implemented by wrapping [collate](https://github.com/dssg/triage/tree/config_doc/src/triage/component/collate). Most terminology here is taken directly from [collate](https://github.com/dssg/triage/tree/config_doc/src/triage/component/collate).

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
    - `from_obj`: from_obj is usually a source table but can be an expression, such as a join (ie ```"cool_stuff join other_stuff using (stuff_id)"```)
    - `knowledge_date_column`: The date column to use for specifying which records to include in temporal features. It is important that the column used specifies the date at which the event is known about, which may be different from the date the event happened.
    - `aggregates_imputation`: top-level imputation rules that will apply to all aggregates functions can also specify `categoricals_imputation` or `array_categoricals_imputation`. You must specified at least one of the top-level or feature-level imputation to cover ever feature being defined.
        - `all`: The `all` rule will apply to all aggregation functions, unless overridden by more specific one
            - `type`: every imputation rule must have a `type` parameter, while some (like 'constant') have other required parameters (`value` here)
            - `value`
        - `max`: specifying `max` here will take precedence over the `all` rule for aggregations using a MAX() function
    - `aggregates`: aggregates and categoricals define the actual features created. So at least one is required. Aggregates of numerical columns.
        - (First quantity)
            - `quantity`: Each quantity is a number of some and the list of metrics are applied to each quantity.
            - `imputation`:  Imputation rules specified at the level of specific features will take precedence over the higer-level rules specified above. Note that the 'count' and 'sum' metrics will be imputed differently here.
                - `count`:
                    - `type`: `mean`
                - `sum`:
                    - `type`: `constant`
                    - `value`: `137`
            - `metrics`:
                - `count`
                - `sum`
            - `coltype`: `smallint` (Optional, if you want to control the column type in the generated features tables)
        - (Second quantity)
            - `quantity`: `some_flag` (Since we're specifying `aggregates_imputation` above, a feature-specific imputation rule can be omitted)
            - `metrics`: 
                - `max`
                - `sum`

    - `categoricals`:  Categorical features. The column given can be of any type, but the choices must comparable to that type for equality within SQL The result will be one feature for each choice/metric combination.
    - (First column)
        - `column`: Note that we haven't specified a top level `categoricals_imputation` set of rules, so we have to include feature-specific imputation rules for both of our categoricals here.
        - `imputation`:
            - `sum`:
                - `type`: `null_category`
            - `max`:
                - `type`: `mean`
        - `choices`:
        - `metrics`: `sum`
    - (Second column)
        - `column`: `shape` (As with the top-level imputation rules, `all` can be used for the feature-level rules to specify the same type of imputation for all aggregation functions)
        - `imputation`:
            - `all`:
                `type`: `zero`
        - `choice_query`: `select distinct shape from cool stuff`
        - `metrics`: 
            - `sum`
    - `intervals`: The time intervals over which to aggregate features
    - `groups`: A list of different columns to separately group by

### Feature Grouping
define how to group features and generate combinations

- `feature_group_definition`: feature_group_definition allows you to create groups/subset of your features by different criteria. for instance,
    - `tables`: allows you to send a list of collate feature tables (collate builds these by appending `aggregation_imputed` to the prefix)
    - `prefix`: allows you to specify a list of feature name prefixes

- `feature_group_strategies`: strategies for generating combinations of groups. available: all, leave-one-out, leave-one-in, all-combinations

### User Metadata
These are arbitrary keys/values that you can have Triage apply to the metadata for every matrix in the experiment. Any keys you include here can be used in the 'model_group_keys' below. For example, if you run experiments that share a temporal configuration but that use different label definitions (say, labeling building inspections with **any** violation as positive or labeling only building inspections with severe health and safety violations as positive), you can use the user metadata keys to indicate that the matrices from these experiments have different labeling criteria. The matrices from the two experiments will have different filenames (and not be overwritten or inappropriately reused), and if you add the label_definition key to the model group keys, models made on different label definition will have different groups. In this way, user metadata can be used to expand Triage beyond its explicitly supported functionality.

- `user_metadata`: `'severe_violations'`

### Model Grouping (optional)
Model groups are a way of partitioning trained models in a way that makes for easier analysis.

`model_group_keys` defines a list of training matrix metadata and classifier keys that should be considered when creating a model group.

There is an extensive default configuration, which is aimed at producing groups whose constituent models are equivalent to each other in all ways except for when they were trained. This makes the analysis of model stability easier.

To accomplish this, the following default keys are used: `class_path`, `parameters`, `feature_names`, `feature_groups`, `cohort_name`, `state`, `label_name`, `label_timespan`, `as_of_date_frequency`, `max_training_history`

If you want to override this list, you can supply a `model_group_keys` value. All of the defaults are available, along with some other temporal information that could be useful for more specialized analyses:

`first_as_of_time`, `last_as_of_time`, `matrix_info_end_time`, `as_of_times`, `feature_start_time`

You can also use any pieces of user_metadata that you included in this experiment definition, as they will be present in the matrix metadata. 
- `model_group_keys`: [`feature_groups`, `label_definition`]

### Grid Configuration
The classifier/hyperparameter combinations that should be trained

Each top-level key should be a class name, importable from triage. sklearn is available, and if you have another classifier package you would like available, contribute it to requirement/main.txt

- `grid_config`: Each lower-level key is a hyperparameter name for the given classifier, and each value is a list of potential values. All possible combinations of classifiers and hyperparameters are trained. Please check out the [grid_config session](https://github.com/dssg/triage/blob/6cb43f9cca032a980cbc25a9501e9559135fd04d/example/config/experiment.yaml#L276) in `experiment.yaml` as for a detailed example.


### Prediction
How predictions are computed for train and test matrices?

- `prediction`: Rank tiebreaking - In the predictions.rank_abs and rank_pct columns, ties in the score are broken either at random or based on the `worst` or `best` options. `worst` is the default.


`worst` will break ties with the ascending label value, so if you take the top **k** predictions, and there are ties across the **k** threshold, the predictions above the threshold will be negative labels if possible.

`best` will break ties with the descending label value, so if you take the top **k** predictions, and there are ties across the **k** threshold, the predictions above the threshold will be positive labels if possible.

`random` will choose one random ordering to break ties. The result will be affected by current state of Postgres' random number generator. Before ranking, the generator is seeded based on the **model**'s random seed.


### Model Scoring
How each trained model is scored?

Each entry in `testing_metric_groups` needs a list of one of the metrics defined in catwalk.evaluation.ModelEvaluator.available_metrics (contributions welcome!) Depending on the metric, either thresholds or parameters.

`Parameters`: specify any hyperparameters needed. For most metrics, which are simply wrappers of sklearn functions, these are passed directly to sklearn. 

- `thresholds` are more specific: The list is dichotomized and only the top percentile or top n entities are scored as positive labels

subsets, if passed, will add evaluations for subset(s) of the predictions to the subset_evaluations tables, using the same testing and training metric groups as used for overall evaluations but with any thresholds reapplied only to entities in the subset on the relevant as_of_dates. For example, when calculating **precision@5_pct** for the subset of women, the ModelEvaluator will count as positively labeled the top 5% of women, rather than any women in the top 5% overall. This is useful if, for example, different interventions will be applied to different subsets of entities (e.g., one program will provide subsidies to the top 500 women with children and another program will provide shelter to the top 150 women without children) and you would like to see whether a single model can be used for both applications. Subsets can also be used to see how a model's performance would be affected if the requirements for intervention eligibility became more restricted.

### Individual Importances
How feature importances for individuals should be computed. This entire section can be left blank, in which case the defaults will be used.

- `individual_importance`:
    - `methods`: Refer to *how to compute* individual importances. Each entry in this list should represent a different method. Available methods are in the catwalk library's: `catwalk.individual_importance.CALCULATE_STRATEGIES` list. Will default to `uniform`, or just the global importances. Empty list means don't calculate individual importances.
    - `n_ranks`: The number of top features per individual to compute importances for. Will default to 5.


## Audition Configuration
Also check out the example file `audition.yaml`.

### Choose Model Groups
Audtion needs a buch of `model_group_id's to help users select the models.

- `model_groups`:
    - `query`: The query is to choose what the model groups you want to include in the first round of model selection.


### Choose Timestamps/Train end times
The timestamps when audition happens for each model group.

- `time_stamps`:
    - `query`: There's a hard rule in Audition that all of the chosen model groups for audition should have the same train end times as the timestamps or the subset of the timestamps from this query, otherwise those model groups with unmatched train end times will be pruned in the first round.


### Filter
Configuration for the Auditioner

- `filter`:
    - `metric`: metric of interest, e.g. `precision@`
    - `parameter`: parameter of interest, e.g. `50_abs`
    - `max_from_best`: The maximum value that the given metric can be worse than the best model for a given train end time.
    - `threshold_value`: The worst absolute value that the given metric should be. 
    - `distance_table`: name of the distance table.
    - `models_table`: name of the models table.

### Rules
The selection rules for Audition to simulate the model selection process for each timestamps.

- `rules`:
    - `shared_parameters`:
        - `metric`: The metric and parameter in shared_parameters have to be the same in the `Filter` section.
        - `parameter`: The metric and parameter in shared_parameters have to be the same in the `Filter` section.
    - `selection_rules`: Rules for selecting the best model. All supported rules can be found in the [Audtion's README](https://github.com/dssg/triage/tree/master/src/triage/component/audition).
