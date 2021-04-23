## Util for Saving Predictions of Selected Model Groups

The script: `predictions_selected_model_groups.py`

config file example: `add_predictions_example_config.py`

This utility accepts a list of model group ids and an experiment hash, and generates & saves test predictions for the relevant models. This util is primarily targetted to be used post-audition for the following use-case:
1. A large model grid was used in the experiment with `save_predictions=False`
2. Audition was used to narrow down model_groups of interest & predictions are required for postmodeling

**Usage**

`python predictions_selected_model_groups.py -c <config_file> -d <database_credentials_file>`

The `<config_file>` is a required argument. If the `<database_credentials_file>` is not provided, the working directory should contain a `database.yaml`.


