# Upgrading your experiment configuration to v8


This document details the steps needed to update a triage v6 configuration to
v8, mimicking the old behavior.

Experiment configuration v8 includes only one change from v7: the `groups` key is no longer supported in the feature configuration (all features must be grouped only at the `entity_id` level).

Old:
```yaml

config_version: 'v7'

# FEATURE GENERATION
feature_aggregations:
  -
    prefix: 'inspections'
    from_obj: 'semantic.events'
    knowledge_date_column: 'date'

    aggregates_imputation:
      count:
        type: 'zero_noflag'

    aggregates:
      -
        quantity:
          total: "*"
        metrics:
          - 'count'

    intervals: ['all']

    groups:
      - 'entity_id'
```

New:
```yaml

config_version: 'v8'

# FEATURE GENERATION
feature_aggregations:
  -
    prefix: 'inspections'
    from_obj: 'semantic.events'
    knowledge_date_column: 'date'

    aggregates_imputation:
      count:
        type: 'zero_noflag'

    aggregates:
      -
        quantity:
          total: "*"
        metrics:
          - 'count'

    intervals: ['all']
```
