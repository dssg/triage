/*
Default query for model_groups using triage results-schema

Assumptions:
- Schema search order specified in db_config
 */

SELECT
    model_group_id,
    model_type,
    model_parameters,
    feature_list,
    model_config
FROM model_groups
WHERE (:no_model_group_subset OR model_group_id IN (:model_group_ids));
