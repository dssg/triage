/*
Default query for feature_importances using triage results-schema

Assumptions:
- Schema search order specified in db_config
- Each model_id has a unique associated model_group_id
- models.train_end_time corresponds to predictions.as_of_date
 */

SELECT
    fimp.model_id,
    mods.model_group_id,
    mods.train_end_time AS as_of_date,
    fimp.feature,
    fimp.feature_importance,
    fimp.rank_abs,
    fimp.rank_pct,
FROM feature_importances fimp
INNER JOIN model_ids mods
ON mods.model_id = fimp.model_id
WHERE mods.train_end_time >= {start_date}
AND mods.train_end_time <= {end_date}
AND ({no_model_group_subset} OR model_group_id IN
    (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
AND ({no_model_id_subset} OR model_id IN
    (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))));
