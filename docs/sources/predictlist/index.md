# Retrain and Predict
Use an existing model group to retrain a new model on all the data up to the current date and then predict forward into the future.

## Examples
Both examples assume you have already run a Triage Experiment in the past, and know these two pieces of information:
1. A `model_group_id` from a Triage model group that you want to use to retrain a model and generate prediction
2. A `today` to generate your predictions on.

### CLI
`triage retrainpredict <model_group_id> <today>`

Example:
`triage retrainpredict 30 2021-04-04`

The `retrainpredict` will assume the current path to be the 'project path' to train models and write matrices, but this can be overridden by sending the `--project-path` option

### Python
The `Retrainer` class from `triage.predictlist` module can be used to retrain a model and predict forward.

```python
from triage.predictlist import Retrainer
from triage import create_engine

retrainer = Retrainer(
    db_engine=create_engine(<your-db-info>),
    project_path='/home/you/triage/project2'
    model_group_id=36,
)
retrainer.retrain(today='2021-04-04')
retrainer.predict(today='2021-04-04')
```

## Output
The retrained model is sotred similariy to the matrices created during an Experiment:
- Raw Matrix saved to the matrices directory in project storage
- Raw Model saved to the trained_model directory in project storage
- Retrained Model info saved in a table (triage_metadata.models) where model_comment = 'retrain_2021-04-04'
- Predictions saved in a table (triage_production.predictions)
- Prediction metadata (tiebreaking, random seed) saved in a table (triage_produciton.prediction_metadata)


# Predictlist
If you would like to generate a list of predictions on already-trained Triage model with new data, you can use the 'Predictlist' module.

# Predict Foward with Existed Model
Use an existing model object to generate predictions on new data.

## Examples
Both examples assume you have already run a Triage Experiment in the past, and know these two pieces of information:
1. A `model_id` from a Triage model that you want to use to generate predictions
2. An `as_of_date` to generate your predictions on.

### CLI
`triage predictlist <model_id> <as_of_date>`

Example:
`triage predictlist 46 2019-05-06`

The predictlist will assume the current path to be the 'project path' to find models and write matrices, but this can be overridden by sending the `--project-path` option.

### Python

The `predict_forward_with_existed_model` function from the `triage.predictlist` module can be used similarly to the CLI, with the addition of the database engine and project storage as inputs.
```
from triage.predictlist import generate predict_forward_with_existed_model 
from triage import create_engine

predict_forward_with_existed_model(
    db_engine=create_engine(<your-db-info>),
    project_path='/home/you/triage/project2'
    model_id=46,
    as_of_date='2019-05-06'
)
```

## Output
The Predictlist is stored similarly to the matrices created during an Experiment:
- Raw Matrix saved to the matrices directory in project storage
- Predictions saved in a table (triage_production.predictions)
- Prediction metadata (tiebreaking, random seed) saved in a table (triage_production.prediction_metadata)

## Notes
- The cohort and features for the Predictlist are all inferred from the Experiment that trained the given model_id (as defined by the experiment_models table).
- The feature list ensures that imputation flag columns are present for any columns that either needed to be imputed in the training process, or that needed to be imputed in the predictlist dataset.


