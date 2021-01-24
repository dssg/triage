## Audition Configuration
`audition.yaml` controls Audition, Triage's model selection module. An example `audition.yaml` can be found [here](https://github.com/dssg/triage/blob/master/example/config/audition.yaml)

### Syntax


```yaml
model_groups:
    query: 'string'

time_stamps:
    query: 'string'

filter:
    metric: 'string'
    parameter: 'string'
    max_from_best: 1.0
    threshold_value: 0.0
    distance_table: 'string'
    models_table: 'models'

rules:
    - shared_parameters:
        - 
            metric: 'string'
            parameter': 'string'
      selection_rules:
        - 
            name: 'string'
            n: 1
```

### Parameters

- `model_groups` (dict):
    - `query` (string): A sql query returning a list of model groups to be included in the first round of model selection
  - `time_stamps` (dict):
    - `query` (string): A sql query specifying the list of train end times over which the specified model groups are evaluated
  - `filter` (dict):
    - `metric` (string): Metric of interest, e.g. `precision@`
    - `parameter` (string): Parameter to metric of interest, e.g. `50_abs`
    - `max_from_best` (float): Model groups that perform this much worse than the best-performing model in at least one train period will be pruned.
    - `threshold_value` (float): Model groups that perform worse than this value in at least one train/test period will be pruned.
    - `distance_table` (string): Name of the table that will store model distance results (created by Triage if not found). 
    - `models_table` (string): Name of the table that stores trained model groups to be evaluated.
    - `agg_type` (string): optional for aggregating metric values across multiple models for a given `model_group_id` and `train_end_time` combination (e.g., from different random seeds) -- `mean`, `best`, or `worst` (the default)
  - `rules` (list): A list of selection rule groups
    - (dict):
        - `shared_parameters` (list): A list of parameters shared by the selection rules in a group
            - (dict): A bundle of shared parameters
                - `metric` (string): Metric of interest, e.g. `precision@`. Should match specification in `filter`.
                - `parameter` (string): Parameter to metric of interest, e.g. `50_abs`. Should match specification in `filter`.
        - `selection_rules` (list): A list of selection rules to be applied
            - (dict): Specification of a [selection rule function](../selection_rules/#selection-rules), including its name and parameters
                - `name` (string): The name of a selection rule function
                - Pass other arguments required by the function here as key:value pairs. Optionally, pass multiple arguments in a list.