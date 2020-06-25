## Selection Rules

Triage supports 8 model selection rules. Each is represented internally by one of the following functions.

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

Each `add_rule_` method takes parameters mapping to the parameters of the corresponding model selection function.


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