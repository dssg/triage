# Risklist

If you would like to generate a list of predictions on already-trained Triage model with new data, you can use the 'Risklist' module.

## Examples
Both examples assume you have already run a Triage Experiment in the past, and know these two pieces of information:
1. A `model_id` from a Triage model that you want to use to generate predictions
2. An `as_of_date` to generate your predictions on.

### CLI
`triage risklist <model_id> <as_of_date>`

Example:
`triage risklist 46 2019-05-06`

The risklist will assume the current path to be the 'project path' to find models and write matrices, but this can be overridden by sending the `--project-path` option.

### Python

The `generate_risk_list` function from the `triage.risklist` module can be used similarly to the CLI, with the addition of the database engine and project storage as inputs.
```
from triage.risklist generate generate_risk_list
from triage.catwalk.component.storage import ProjectStorage
from triage import create_engine

generate_risk_list(
    db_engine=create_engine(<your-db-info>),
    project_storage=ProjectStorage('/home/you/triage/project2')
    model_id=46,
    as_of_date='2019-05-06'
)
```

## Output
The Risklist is stored similarly to the matrices created during an Experiment:
- Raw Matrix saved to the matrices directory in project storage
- Predictions saved in a table (production.list_predictions)
- Prediction metadata (tiebreaking, random seed) saved in a table (production.prediction_metadata)

## Notes
- The cohort and features for the Risklist are all inferred from the Experiment that trained the given model_id (as defined by the experiment_models table).
- The feature list ensures that imputation flag columns are present for any columns that either needed to be imputed in the training process, or that needed to be imputed in the risklist dataset.
