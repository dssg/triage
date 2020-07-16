## Selection Rules

The Triage uses *selection rules* to compare the performance of trained model groups over time, and select a model group for future predictions. A selection rule tries to predict the best-performing model group in some train/test period, based on the historical performance of each model group on some metric.

For example, a simple selection rule might predict that the best-performing model group during one train/test period will perform best in the following period.

A selection rule can be evaluated by calculating its *regret*, or the difference between the performance of its selected model group and the best-performing model group in some period.

Triage supports 8 model selection rules. Each is represented internally by one of the following functions:

::: triage.component.audition.selection_rules
    rendering:
        heading_level: 3
        show_root_toc_entry: False
    selection:
        filters: 
            - "!^BoundSelectionRule"
            - "!^_"

## RuleMakers

Triage uses `RuleMaker` classes to conveniently format the parameter grids accepted by `make_selection_rule_grid`. Each type of `RuleMaker` class holds methods that build parameter grids for a subset of the available selection rules.

The arguments of each `add_rule_` method map to the arguments of the corresponding model selection function.


::: triage.component.audition.rules_maker
    rendering:
        show_if_no_docstring: True
        show_category_heading: False
        show_root_heading: False
        show_root_toc_entry: False
        heading_level: 3
    selection:
        members:
            - SimpleRuleMaker
            - TwoMetricsRuleMaker
            - RandomGroupRuleMaker
  
## Selection Grid

::: triage.component.audition.selection_rule_grid
    rendering:
        heading_level: 3
        show_root_toc_entry: False