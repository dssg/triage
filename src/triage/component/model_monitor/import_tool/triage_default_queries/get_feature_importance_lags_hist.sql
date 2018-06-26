WITH date_lags AS (
    SELECT
        dts.as_of_date,
        (dts.as_of_date::DATE - '{compare_interval}'::INTERVAL)::DATE AS as_of_date_lag
    FROM (
        SELECT DISTINCT preds.as_of_date
        FROM predictions preds
        INNER JOIN models mods
        ON mods.model_id = preds.model_id
        WHERE  preds.as_of_date <= '{end_date}'
        AND preds.as_of_date >= '{start_date}'
        AND ({no_model_group_subset} OR mods.model_group_id IN
            (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
        AND ({no_model_id_subset} OR mods.model_id IN
            (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))))
    ) dts
), valid_date_lags AS (
    SELECT
        dl1.as_of_date,
        dl1.as_of_date_lag
    FROM date_lags dl1
    INNER JOIN date_lags dl2
    ON dl1.as_of_date_lag = dl2.as_of_date
), feature_lags AS (
    SELECT
        mods.train_end_time::DATE AS as_of_date_lag,
        feats.model_id,
        mods.model_group_id,
        feats.feature,
        feats.feature_importance AS feature_importance_lag,
        feats.rank_abs AS rank_abs_lag,
        feats.rank_pct AS rank_pct_lag
    FROM feature_importances feats
    INNER JOIN models mods
    ON feats.model_id = mods.model_id
    INNER JOIN valid_date_lags vgl
    ON vgl.as_of_date_lag = mods.train_end_time::DATE
    WHERE ({no_model_group_subset} OR mods.model_group_id IN
        (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
    AND ({no_model_id_subset} OR mods.model_id IN
        (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))))
)
SELECT
    vdl.as_of_date::DATE AS as_of_date,
    vdl.as_of_date_lag::DATE AS as_of_date_lag,
    feats.model_id,
    mods.model_group_id,
    feats.feature,
    feats.feature_importance,
    feats.rank_abs,
    feats.rank_pct,
    flag.feature_importance_lag,
    flag.rank_abs_lag,
    flag.rank_pct_lag
FROM feature_importances feats
INNER JOIN models mods
ON mods.model_id = feats.model_id
INNER JOIN valid_date_lags vdl
ON vdl.as_of_date = mods.train_end_time::DATE
INNER JOIN feature_lags flag
ON flag.as_of_date_lag = vdl.as_of_date_lag
AND flag.feature = feats.feature
AND mods.model_group_id = flag.model_group_id
WHERE ({no_model_group_subset} OR mods.model_group_id IN
    (SELECT(UNNEST(ARRAY{model_group_ids}::INTEGER[]))))
AND ({no_model_id_subset} OR mods.model_id IN
    (SELECT(UNNEST(ARRAY{model_ids}::INTEGER[]))))
