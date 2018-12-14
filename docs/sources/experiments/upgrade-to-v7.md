# Upgrading your experiment configuration to v7


This document details the steps needed to update a triage v6 configuration to
v7, mimicking the old behavior.

Experiment configuration v7 includes only one change from v6: The features are given at a different key. Instead of `feature_aggregations`, to make space for non-collate features to be added in the future, there is now a more generic `features` key, under which collate features reside at `spacetime_aggregations`.


Old:

```
feature_aggregations:
    -
        prefix: 'prefix'
        from_obj: 'cool_stuff'
        knowledge_date_column: 'open_date'
        aggregates_imputation:
            all:
                type: 'constant'
                value: 0
        aggregates:
            -
                quantity: 'homeless::INT'
                metrics: ['count', 'sum']
        intervals: ['1 year', '2 year']
        groups: ['entity_id']
```

New:

```
features:
    spacetime_aggregations:
        -
            prefix: 'prefix'
            from_obj: 'cool_stuff'
            knowledge_date_column: 'open_date'
            aggregates_imputation:
                all:
                    type: 'constant'
                    value: 0
            aggregates:
                -
                    quantity: 'homeless::INT'
                    metrics: ['count', 'sum']
            intervals: ['1 year', '2 year']
            groups: ['entity_id']
```

## Upgrading the experiment config version

At this point, you should be able to bump the top-level experiment config version to v7:

Old:

```
config_version: 'v6'
```

New:

```
config_version: 'v7'
```

