/*
Default query for explicit prediction lags
ASSUMES model_ids vary within group ONLY by training as_of_date
 */

SELECT
    '{end_date}'::DATE AS as_of_date,
    ('{end_date}'::DATE - '{compare_interval}'::INTERVAL)::DATE AS as_of_date_lag,
    preds.model_id::INT,
    mods.model_group_id::INT,
    preds.entity_id::INT,
    preds.score::FLOAT4,
    preds.label_value::INT,
    preds.rank_abs::INT,
    preds.rank_pct::FLOAT4,
    lag_preds.score_lag::FLOAT4,
    lag_preds.label_value_lag::INT,
    lag_preds.rank_abs_lag::INT,
    lag_preds.rank_pct_lag::FLOAT4
FROM predictions preds
INNER JOIN models mods
ON mods.model_id = preds.model_id
INNER JOIN (
    SELECT
        lpreds.model_id,
        lmods.model_group_id,
        lpreds.entity_id,
        lpreds.score AS score_lag,
        lpreds.label_value AS label_value_lag,
        lpreds.rank_abs AS rank_abs_lag,
        lpreds.rank_pct AS rank_pct_lag
    FROM predictions lpreds
    INNER JOIN models lmods
    ON lmods.model_id = lpreds.model_id
    WHERE lpreds.as_of_date = ('{end_date}'::DATE - '{compare_interval}'::INTERVAL)::DATE
    AND ({no_model_group_subset} OR model_group_id IN
        (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
    AND ({no_model_id_subset} OR lmods.model_id IN
        (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))))
) lag_preds
ON lag_preds.entity_id = preds.entity_id
AND lag_preds.model_group_id = mods.model_group_id
WHERE preds.as_of_date = '{end_date}'
AND ({no_model_group_subset} OR mods.model_group_id IN
    (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
AND ({no_model_id_subset} OR mods.model_id IN
    (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[])))) ;
