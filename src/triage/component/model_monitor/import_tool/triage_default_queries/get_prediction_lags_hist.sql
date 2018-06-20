CREATE OR REPLACE FUNCTION model_monitor.get_prediction_lags_hist(_start_date DATE,
																  _end_date DATE,
																  _lag_interval INTERVAL,
																  _model_ids TEXT,
																  _no_model_id_subset BOOLEAN,
																  _model_group_ids TEXT,
																  _no_model_id_subset BOOLEAN))
RETURNS TABLE (
	as_of_date DATE,
	as_of_date_lag DATE,
	model_id INT,
	model_group_id INT,
	entity_id INT,
	score FLOAT4,
	label_value INT,
	rank_abs INT,
	rank_pct FLOAT4,
	score_lag FLOAT4,
	label_value_lag INT,
	rank_abs_lag INT,
	rank_pct_lag FLOAT4
) AS
$BODY$
BEGIN
	RETURN QUERY
	WITH date_lags AS (
		SELECT
			dts.as_of_date,
			dts.as_of_date - _lag_interval AS as_of_date_lag
		FROM (
			SELECT DISTINCT preds.as_of_date
			FROM predictions preds
			INNER JOIN models mods
			ON mods.model_id = preds.model_id
			WHERE  preds.as_of_date <= _end_date
			AND preds.as_of_date >= _start_date
			AND (_no_model_group_subset OR mods.model_group_id IN (string_to_array(_model_group_ids)))
            AND (_no_model_id_subset OR preds.model_id IN (string_to_array(_model_ids)));
		) dts
	), valid_date_lags AS (
		SELECT
			dl1.as_of_date,
			dl1.as_of_date_lag
		FROM date_lags dl1
		INNER JOIN date_lags dl2
		ON dl1.as_of_date_lag = dl2.as_of_date
	), prediction_lags AS (
		SELECT
			preds.as_of_date AS as_of_date_lag,
			preds.model_id,
			preds.entity_id,
			preds.score AS score_lag,
			preds.label_value AS label_value_lag,
			preds.rank_abs AS rank_abs_lag,
			preds.rank_pct AS rank_pct_lag
		FROM predictions preds
		INNER JOIN valid_date_lags vgl
		ON vgl.as_of_date_lag = preds.as_of_date
		INNER JOIN models mods
		ON preds.model_id = mods.model_id
		WHERE (_no_model_group_subset OR mods.model_group_id IN (string_to_array(_model_group_ids)))
        AND (_no_model_id_subset OR preds.model_id IN (string_to_array(_model_ids)));
	)
	SELECT
		vdl.as_of_date::DATE as as_of_date,
		vdl.as_of_date_lag::DATE as as_of_date_lag,
		preds.model_id::INT,
		mods.model_group_id::INT,
		preds.entity_id::INT,
		preds.score::FLOAT4,
		preds.label_value::INT,
		preds.rank_abs::INT,
		preds.rank_pct::FLOAT4,
		plag.score_lag::FLOAT4,
		plag.label_value_lag::INT,
		plag.rank_abs_lag::INT,
		plag.rank_pct_lag::FLOAT4
	FROM predictions preds
	INNER JOIN valid_date_lags vdl
	ON vdl.as_of_date = preds.as_of_date
	INNER JOIN models mods
	ON mods.model_id = preds.model_id
	INNER JOIN prediction_lags plag
	ON plag.as_of_date_lag = vdl.as_of_date_lag
	AND plag.entity_id = preds.entity_id
	AND (_no_model_group_subset OR mods.model_group_id IN (string_to_array(_model_group_ids)))
    AND (_no_model_id_subset OR preds.model_id IN (string_to_array(_model_ids)));
END
$BODY$
LANGUAGE plpgsql VOLATILE
COST 100;
