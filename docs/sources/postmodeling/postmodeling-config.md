## Postmodeling Configuration

The Triage Postmodeling module is controlled by two config files: `postmodeling_config.yaml` and `postmodeling_crosstabs.yaml`.

### Postmodeling Configuration File
Configuration for the Triage Postmodeling module. An example `postmodeling_config.yaml` file can be found [here](https://github.com/dssg/triage/blob/master/example/config/postmodeling_config.yaml).

- `project_path`: Project path defined in triage with matrices and models
- `audition_output_path`: Audition output path
- `model_group_id`: List of model_id's [optional if a audition_output_path is given]
- `thresholds`: Thresholds for defining positive predictions
- `baseline_query`: SQL query for defining a baseline for comparison in plots. It needs a metric and parameter
- `max_depth_error_tree`: For error trees, how depth the decision trees should go?
- `n_features_plots`: Number of features for importances
- `figsize`: Default size for plots
- `fontsize`: Default fontsize for plots


### Postmodeling Crosstabs Configuration File
Configuration for crosstabs in Triage's Postmodeling module. An example `postmodeling_crosstabs.yaml` file can be found [here](https://github.com/dssg/triage/blob/master/example/config/postmodeling_crosstabs.yaml).

- `output`: Define the schema and table for crosstabs
- `thresholds`: Thresholds for defining positive predictions
- `entity_id_list`: (optional) a list of `entity_ids` to subset on the crosstabs analysis
- `models_list_query`: SQL query for getting `model_id`s
- `as_of_dates_query`: SQL query for getting `as_of_date`s
- `models_dates_join_query`: don't change the default query unless strictly necessary. It is just validating pairs of (`model_id`, `as_of_date`) in a predictions table
- `features_query`: features_query must join `models_dates_join_query` with 1 or more features table using `as_of_date`
- `predictions_query`: the predictions query must return `model_id`, `as_of_date`, `entity_id`, `score`, `label_value`, `rank_abs` and `rank_pct`. It must join `models_dates_join_query` using both `model_id` and `as_of_date`. 

