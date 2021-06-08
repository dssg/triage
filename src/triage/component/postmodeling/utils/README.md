## Util for Saving Predictions of Selected Model Groups

The script: `predictions_selected_model_groups.py`

config file example: `add_predictions_example_config.py`

This utility accepts a list of model group ids and an experiment hash, and generates & saves test predictions for the relevant models. This util is primarily targetted to be used post-audition for the following use-case:
1. A large model grid was used in the experiment with `save_predictions=False`
2. Audition was used to narrow down model_groups of interest & predictions are required for postmodeling

### Usage

**Command Line Inerface**

This util is available with the triage CLI and can be run using the following command

`triage [-d <database_credentials_file>] savepredictions -c <config_file> `

The `<config_file>` is a required argument. If the `<database_credentials_file>` is not provided, the working directory should contain a `database.yaml`.


**Python Interface**

The function can be imported into a python script to add predictions of selected model groups as shown below. The parameters could be read

    from src.triage.postmodeling.utils.predictions_selected_model_groups import generate_predictions

    generate_predictions(
        db_engine=conn, # The database connection
        model_groups=[1, 2, 3], # List of model groups  
        project_path='path/to/models/and/matrices', # where the models and matrices are stored
        experiment_hashes=[
            'fdb2ee5499b30d53048a7253cc5be36f', 
            'b045af84cd13830f9b43ce5125ecac0b'
        ], # Restricting models (in the above model groups) based on exeriment (optional)
        train_end_times_range={
            'range_start_date': '2015-01-01',
            'range_end_date': '2017-01-01'
        }, # Restricing models based on train end times (optional). Intervals are inclusive and can be open ended. 
        rank_order='worst', # How to break ties
        replace=True # Whether to replace existing predictions
    )


