# Upgrading your experiment configuration to v6


This document details the steps needed to update a triage v5 configuration to
v6, mimicking the old behavior.

Experiment configuration v6 includes only one change from v5: When specifying
the `cohort_config`, if a `query` is given , the `{af_of_date}` is no longer
quoted or casted by Triage. Instead, the user must perform the quoting and
casting, as is done already for the `label_config`.

Old:

```
cohort_config:
    query: |
        SELECT DISTINCT entity_id
          FROM semantic.events
         WHERE event = 'booking'
           AND startdt <@ daterange(({as_of_date} - '3 years'::interval)::date, {as_of_date})
           AND enddt < {as_of_date}
         LIMIT 100
    name: 'booking_last_3_years_limit_100'
```

New:

```
cohort_config:
    query: |
        SELECT DISTINCT entity_id
          FROM semantic.events
         WHERE event = 'booking'
           AND startdt <@ daterange(('{as_of_date}'::date - '3 years'::interval)::date, '{as_of_date}'::date)
           AND enddt < '{as_of_date}'
         LIMIT 100
    name: 'booking_last_3_years_limit_100'
```

## Upgrading the experiment config version

At this point, you should be able to bump the top-level experiment config version to v6:

Old:

```
config_version: 'v5'
```

New:

```
config_version: 'v6'
```

