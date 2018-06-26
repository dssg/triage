/*
Default query for models using triage results-schema

Assumptions:
- Schema search order specified in db_config
- Each model_id has a unique associated model_group_id
 */

SELECT
    model_id,
    model_group_id,
    model_parameters,
    model_comment,
    batch_comment,
    config,
    train_end_time,
    test,
    train_label_window
FROM models
WHERE ({no_model_group_subset} OR model_group_id IN
    (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
AND ({no_model_id_subset} OR model_id IN
    (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))));
