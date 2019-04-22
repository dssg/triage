# Upgrading your experiment configuration to v5


This document details the steps needed to update a triage v3 or v4 configuration to v5, mimicing the old behavior (as opposed to taking advantage of new options) as much as possible.

In the experiment configuration v5, several things were changed:

- `state_config` becomes `cohort_config`, and receives new options
- `label_config` is changed to take a parameterized query
- `model_group_keys` is changed to have more robust defaults, and values specified in the config file act as overrides for the defaults instead of additions to them.

## state_config -> cohort_config

Upgrading the state config is fairly straightforward, as no functionality was removed. The key at which the state table-based configuration can be passed has changed. Before it resided at the top-level `state_config` key, whereas now it is in the optional `dense_states` key within the top-level `cohort_config` key.

Old:

```
state_config:
    table_name: 'states'
    state_filters:
        - 'state_one AND state_two'
        - '(state_one OR state_two) AND state_three'
```

New:

```
cohort_config:
   dense_states:
        table_name: 'states'
        state_filters:
        - 'state_one AND state_two'
        - '(state_one OR state_two) AND state_three'
```

## label_config

The label config has had functionality changed, so there is more conversion that needs to happen. Instead of taking in an 'events' table and making assumptions suitable for inspections tasks based on that table, for transparency and flexibility this now takes a parameterized query, as well as an optional `include_missing_labels_in_train_as` boolean. Leaving out this boolean value reproduces the inspections behavior (missing labels are treated as null), so to upgrade old configurations it is not needed.

Old:

```
events_table: 'events'
```

New:

```
label_config:
    query: |
        select
        events.entity_id,
        bool_or(outcome::bool)::integer as outcome
        from events
        where '{as_of_date}' <= outcome_date
            and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
            group by entity_id
```

## model_group_keys

The model group configuration was changed quite a bit. Before, the Experiment defined a few default grouping keys and would treat anything included in the config as additional. In practice, there were many keys that were almost always included as additional model group keys, and these are now default. There are also other keys that generally make sense if certain things are iterated on (e.g. feature groups). The goal is for most projects to simply leave out this configuration value entirely. If possible, this is the recommended route to go. But for the purposes of this guide, this change should duplicate the old behavior exactly.

Old (empty, using defaults):

```
```

New:

```
model_group_keys: ['class_path', 'parameters', 'feature_names']
```

Old (more standard in practice, adding some temporal parameters):

```
model_group_keys: ['label_timespan', 'as_of_date_frequency', 'max_training_history']
```

New:
```
model_group_keys: ['class_path', 'parameters', 'feature_names', 'label_timespan', 'as_of_date_frequency', 'max_training_history']
```

## Upgrading the experiment config version

At this point, you should be able to bump the top-level experiment config version to v5:

Old:

```
config_version: 'v4'
```

New:

```
config_version: 'v5'
```
