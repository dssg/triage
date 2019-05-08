# Upgrading your experiment configuration to v7


This document details the steps needed to update a triage v6 configuration to
v7, mimicking the old behavior.

Experiment configuration v7 includes only one change from v6: The features are given at a different key. Instead of `feature_aggregations`, to make space for non-collate features to be added in the future, there is now a more generic `features` key. The value of this key is a dictionary, the key of which is the desired output table name for that feature table, and the value of which is the same as the configuration for each feature aggregation from before. There is one change to this. A new key called 'feature_generator_type', to specify which method is being used to generate this feature table. Since non-collate features have not been added yet, there is only one key for this: `spacetime_aggregation`. 

Since the output feature table name is now configurable, there are two things to note:
- Final tables won't necessarily be suffixed with `_aggregation_imputed` as they were before. If you would like to use the old naming system, for instance to avoid having to change postmodeling code that reads features from the database, you can add that suffix to your table name. The example below does set the table name to match what it was before, but there's no reason you have to follow this if you don't want! You can call the table whatever you want.
- The `prefix` key is no longer used to construct the table name. It is still used to prefix column names, if present. If not present, the name of the feature table will be used.



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
    prefix_aggregation_imputed:
        feature_generator_type: 'spacetime_aggregation'
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

