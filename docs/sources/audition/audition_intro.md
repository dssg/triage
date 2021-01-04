## What is Audition?

Audition is the Triage model selection module. Model selection is the process of selecting a **model group** that is likely to perform well on future data. Audition makes model selection easy and repeatable, even when comparing hundreds of model groups.

Start by defining a set of coarse filtering rules that prune the worst-performing model groups. You can then apply more complex **selection rules** that rank model groups on factors like their performance over time or variability in performance. 

Audition can compare the performance of these selection rules, helping you choose a selection rule that is likely to identify well-performing models in the future.

Audition works well in a **Jupyter notebook**, or any other environment that will conveniently display Audition's matplotlib output.

## The Audition Python interface

### Initializing Auditioner

This Auditioner object reads performance data from a database populated by a Triage Experiment. It loads information about the model groups selected by `model_group_ids`, over the training sets specified by `train_end_times`, and filters them as defined by `initial_metric_filters`. 

```python
from triage.component.audition import Auditioner

aud = Auditioner(
    db_engine = conn, # connection to a database populated by a Triage experiment
    model_group_ids=[i for i in range(101, 150)], # selecting model groups to evaluate
    train_end_times=end_times,
    initial_metric_filters=
    	[{'metric': 'precision@',
    	  'parameter': '50_abs',
    	  'max_from_best': 0.3,
    	  'threshold_value': 0.5}],
    models_table='models',
    distance_table='best_dist',
    agg_type='worst'
)
```
Here, Auditioner drops all models group that meet at least one of these conditions in at least one training set:

- Achieves precision at least 0.3 worse than the best performing model group
- Achieves precision worse than 0.5

Note that the `agg_type` parameter is optional for aggregating metric values across multiple models for a given `model_group_id` and `train_end_time` combination (e.g., from different random seeds) -- `mean`, `best`, or `worst` (the default)

### Selection rules
[Selection rules](../api/audition/selection_rules.md#Selection-Rules) allow you to pare down your model groups even more.

Start by adding rules to a [`RuleMaker` object](../api/audition/selection_rules.md/#RuleMakers):

```python
from triage.component.audition.rules_maker import SimpleRuleMaker, create_selection_grid

rule = SimpleRuleMaker()

rule.add_rule_best_current_value(metric='precision@', parameter='50_abs', n=3)
rule.add_rule_best_average_value(metric='precision@', parameter='50_abs', n=3)
```
These rules select the top 3 models 

- Ranked by precision in the most recent training set
- Ranked by average precision over all training sets.


Create a selection grid from your `RuleMaker`, and register it in your `Auditioner` object. This will generate a set of plots showing the performance of each rule, and allow you to view the top `n` models selected by each rule.

```py
grid = create_selection_grid(rule)

aud.register_selection_rule_grid(grid, plot=True)
aud.selection_rule_model_group_ids
```


### Metric Filters

Auditioner implements two coarse filters for pruning worst-performing model groups.

These filters are defined on the metrics specified by the `metric` and `parameter` arguments in the `initial_metric_filters` dict. The combination of these two arguments should refer to a metric calculated by a Triage experiment.  

**`max_from_best`**: Model groups that perform this much worse than the best-performing model group in any period will be pruned.

**`threshold_value`**: Model groups that perform worse than this threshold in any period will be pruned.

#### Adding metric filters

After defining an `Auditioner` instance, we can output the models permitted by the initial thresholds.

```python
# Output the thresholded model group ids
aud.thresholded_model_group_ids
```

If that didn't thin things out too much, let's get a bit more agressive with both parameters. If we want to have multiple filters, then use `update_metric_filters` to apply a set of filters to the model groups we're considering in order to eliminate poorly performing ones. The model groups will be plotted again after updating the filters.

```python
aud.update_metric_filters([{
    'metric': 'precision@',
    'parameter': '50_abs',
    'max_from_best': 0.5,
    'threshold_value': 0.12
}])
aud.thresholded_model_group_ids
```

## The Audition CLI

Besides its Python interface, Audition exposes a full-featured CLI.

Start by defining an [Audition config file](../api/audition/audition-config.md). Parameters in an Audition config map to arguments in the Python interface introduced above.

```sh
triage -d dbconfig.yaml audition --config audition_config.yaml --directory audition_output
```

This command will run Audition against the database specified in `dbconfig.yaml`, using options from `audition_config.yaml`. It will store the resulting plots in the directory `audition_output`.