## What is Audition?

Audition is the Triage model selection module. It provides a set of tools for selecting a best-performing model from a large set of model groups.

Users first define a set of coarse rules that filter out the worst-performing model groups. Users can then apply more complex selection rules that rank model groups on factors like their performance over time or variability. 

Audition can compare the performance of these selection rules, helping users choose a selection rule that is likely to generate well-performing models.

## Quick Example

This Auditioner object loads performance from the fifty specified model groups, over the training sets specified by `train_end_times`, and filters them as defined by `initial_metric_filters`. 

```python
from triage.component.audition import Auditioner

aud = Auditioner(
    db_engine = conn, # a postgres database connection
    model_group_ids=[i for i in range(101, 150)],
    train_end_times=end_times,
    initial_metric_filters=
    	[{'metric': 'precision@',
    	  'parameter': '50_abs',
    	  'max_from_best': 0.3,
    	  'threshold_value': 0.5}],
    models_table='models',
    distance_table='best_dist'
)
```
Here, Auditioner drops all models group that meet at least one of these conditions in at least one training set:
- Achieves precision at least 0.3 worse than the best performing model group
- Achieves precision worse than 0.5

#### Selection rules
Registering selection rules in an Auditioner object allows you to pare down your model groups even more.

Start by adding rules to a [`RuleMaker` object](selection_rules.md/#RuleMakers):

```python
from triage.component.audition.rules_maker import SimpleRuleMaker, create_selection_grid

rule = SimpleRuleMaker()

rule.add_rule_best_current_value(metric='precision@', parameter='50_abs', n=3)
rule.add_rule_best_average_value(metric='precision@', parameter='50_abs', n=3)
```
These rules select the top 3 models 
- Ranked by precision in the most recent training set
- Ranked by average precision over all training sets.

Create a selection grid from your `RuleMaker`, and register it in your `Auditioner`. This will generate a set of plots showing the performance of each rule, and allow you to view the top `n` models selected by each rule.
```py
grid = create_selection_grid(rule)

aud.register_selection_rule_grid(grid, plot=True)
aud.selection_rule_model_group_ids
```