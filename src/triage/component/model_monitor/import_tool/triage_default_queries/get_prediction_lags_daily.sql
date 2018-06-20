CREATE OR REPLACE FUNCTION model_monitor.get_prediction_lags_daily(_as_of_date DATE,
																   _lag_interval INTERVAL,
																   _model_ids TEXT,
																   _no_model_id_subset BOOLEAN,
																   _model_group_ids TEXT,
																   _no_model_id_subset BOOLEAN)
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
	SELECT
		_as_of_date::DATE AS as_of_date,
		(_as_of_date - _lag_interval)::DATE AS as_of_date_lag,
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
	INNER JOIN
		(SELECT
		    lpreds.model_id,
		    lmods.model_group_id,
			lpreds.entity_id,
			lpreds.score AS score_lag,
			lpreds.label_value AS label_value_lag,
			lpreds.rank_abs AS rank_abs_lag,
			lpreds.rank_pct AS rank_pct_lag
		FROM predictions lpreds
		INNER JOIN models lmods
		ON lmods.model_id = preds.model_id
		WHERE lpreds.as_of_date = _as_of_date - _lag_interval
		AND (_no_model_group_subset OR mods.model_group_id IN (string_to_array(_model_group_ids)))
        AND (_no_model_id_subset OR preds.model_id IN (string_to_array(_model_ids))) lag_preds
	ON lag_preds.entity_id = preds.entity_id
	AND lag_preds.model_id = preds.model_id
	WHERE preds.as_of_date = _as_of_date
	AND (_no_model_group_subset OR mods.model_group_id IN (string_to_array(_model_group_ids)))
    AND (_no_model_id_subset OR preds.model_id IN (string_to_array(_model_ids));
END;
$BODY$
LANGUAGE plpgsql VOLATILE
COST 100;
