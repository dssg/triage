/*
Default query for explicit prediction lags
ASSUMES model_ids vary within group ONLY by training as_of_date
 */

SELECT
    '{end_date}'::DATE AS as_of_date,
    ('{end_date}'::DATE - '{compare_interval}'::INTERVAL)::DATE AS as_of_date_lag,
    feats.model_id,
    mods.model_group_id,
    feats.feature,
    feats.feature_importance,
    feats.rank_abs,
    feats.rank_pct,
    lag_feats.feature_importance_lag,
    lag_feats.rank_abs_lag,
    lag_feats.rank_pct_lag
FROM feature_importances feats
INNER JOIN models mods
ON mods.model_id = feats.model_id
INNER JOIN (
    SELECT
        lfeats.model_id,
        lmods.model_group_id,
        lfeats.feature,
        lfeats.feature_importance AS feature_importance_lag,
        lfeats.rank_abs AS rank_abs_lag,
        lfeats.rank_pct AS rank_pct_lag
    FROM feature_importances lfeats
    INNER JOIN models lmods
    ON lmods.model_id = lfeats.model_id
    WHERE lmods.train_end_time::DATE = ('{end_date}'::DATE - '{compare_interval}'::INTERVAL)::DATE
    AND ({no_model_group_subset} OR model_group_id IN
        (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
    AND ({no_model_id_subset} OR lmods.model_id IN
        (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))))
) lag_feats
ON lag_feats.feature = feats.feature
AND lag_feats.model_group_id = mods.model_group_id
WHERE mods.train_end_time::DATE = '{end_date}'::DATE
AND ({no_model_group_subset} OR mods.model_group_id IN
    (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
AND ({no_model_id_subset} OR mods.model_id IN
    (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[])))) ;
