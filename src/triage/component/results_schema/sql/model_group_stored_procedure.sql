/*
Function for using the model group table. This function requires a table like
-----------
CREATE TABLE triage_metadata.model_groups
(
  model_group_id    SERIAL PRIMARY KEY,
  model_type        TEXT,
  hyperparameters   JSONB,
  feature_list      TEXT []
  model_config		JSONB
);
-----------
populates the table and returns the IDs
*/
CREATE OR REPLACE FUNCTION public.get_model_group_id(in_model_type        TEXT,
                                             in_hyperparameters   JSONB,
                                             in_feature_list      TEXT [],
                                             in_model_config      JSONB)
  RETURNS INTEGER AS
$BODY$
DECLARE
  model_group_return_id INTEGER;
BEGIN
  --Obtain an advisory lock on the table to avoid double execution
  PERFORM pg_advisory_lock(60637);

  -- Check if the model_group_id exists, if not insert the model parameters and return the new value
  SELECT *
  INTO model_group_return_id
  FROM triage_metadata.model_groups
  WHERE
    model_type = in_model_type
    AND hyperparameters = in_hyperparameters
    AND feature_list = ARRAY(Select unnest(in_feature_list) ORDER BY 1)
    AND model_config = in_model_config ;
  IF NOT FOUND
  THEN
    INSERT INTO triage_metadata.model_groups (model_group_id, model_type, hyperparameters, feature_list, model_config)
    VALUES (DEFAULT, in_model_type, in_hyperparameters, ARRAY(Select unnest(in_feature_list) ORDER BY 1), in_model_config)
    RETURNING model_group_id
      INTO model_group_return_id;
  END IF;

  -- Release the lock again
  PERFORM pg_advisory_unlock(60637);


  RETURN model_group_return_id;
END;

$BODY$
LANGUAGE plpgsql VOLATILE
COST 100;



comment on function get_model_group_id (text, jsonb, text [], jsonb) is 'Function for using the model group table. This function requires a table like
-----------
CREATE TABLE triage_metadata.model_groups
(
  model_group_id    SERIAL PRIMARY KEY,
  model_type        TEXT,
  hyperparameters   JSONB,
  feature_list      TEXT []
  model_config		JSONB
);
-----------
populates the table and returns the IDs';
