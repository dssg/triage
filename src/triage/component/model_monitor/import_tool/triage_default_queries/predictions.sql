/*
Default query for predictions using triage results-schema

Assumptions:
- Schema search order specified in db_config
- Each model_id has a unique associated model_group_id
 */

SELECT
    model_id,
    model_group_id,
    as_of_date,
    score,
    label_value,
    rank_abs,
    rank_pct
FROM predictions
WHERE as_of_date >= :start_date
AND as_of_date <= :end_date
AND (:no_model_group_subset OR model_group_id IN (:model_group_ids))
AND (:no_model_id_subset OR model_id IN (:model_ids));
