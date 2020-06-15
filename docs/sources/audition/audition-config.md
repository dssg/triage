## Audition Configuration
`audition.yaml` controls Audition, Triage's model selection module. An example `audition.yaml` can be found [here](https://github.com/dssg/triage/blob/master/example/config/audition.yaml)

### Choose Model Groups
Audtion needs a buch of `model_group_id`s to help users select the models.

- `model_groups`:
    - `query`: The query is to choose what the model groups you want to include in the first round of model selection.


### Choose Timestamps/Train end times
The timestamps when audition happens for each model group.

- `time_stamps`:
    - `query`: There's a hard rule in Audition that all of the chosen model groups for audition should have the same train end times as the timestamps or the subset of the timestamps from this query, otherwise those model groups with missing temstamps will be pruned in the first round.


### Filter
Configuration for the Auditioner

- `filter`:
    - `metric`: metric of interest, e.g. `precision@`
    - `parameter`: parameter of interest, e.g. `50_abs`
    - `max_from_best`: The maximum value that the given metric can be worse than the best model for a given train end time.
    - `threshold_value`: The worst absolute value that the given metric should be. 
    - `distance_table`: name of the distance table that will be created by Auditioner.
    - `models_table`: name of the models table.

### Rules
The selection rules for Audition to simulate the model selection process for each timestamps.

- `rules`:
    - `shared_parameters`:
        - `metric`: The metric and parameter in shared_parameters have to be the same in the `Filter` section.
        - `parameter`: The metric and parameter in shared_parameters have to be the same in the `Filter` section.
    - `selection_rules`: Rules for selecting the best model. All supported rules can be found in the [Audtion's README](https://github.com/dssg/triage/tree/master/src/triage/component/audition).

