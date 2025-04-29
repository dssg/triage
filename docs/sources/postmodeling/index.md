# Postmodeling

The bulk of postmodeling is documented on its README. [Read it here](https://github.com/dssg/triage/tree/master/src/triage/component/postmodeling)

## Crosstabs

One of the features in postmodeling, detailed in the README above, is the ability to view crosstabs. Using the `.crosstabs` property on the `ModelEvaluator` requires the `test_results.crosstabs` table to be created first.  You can do this either with the CLI or in a Python console:

The model crosstabs populates a table in a postgres database containing `model_id`, `as_of_date`, `threshold_unit`, `threshold_value`, and `feature_column`, such as the mean and standard deviation
of the given feature for the predicted-high-risk and predicted-low-risk group. In other words, this table provides simple
descriptives of the feature distributions for the high-risk and low-risk entities, given a specific model and decision
threshold.

The crosstabs config consist of the following parameters:

1. Output schema and table - the name of the schema and table in the postgresdb where the results should be pushed to.

2. Lists of thresholds (abs and/or pct) to serve as cutoff for high risk (positive) and low risk(negative) predictions.

3. (optional) a list of entity_ids to subset for the crosstabs the analysis

4. Models list query must return a column `model_id`. You can pass an explicit array of model ids
using `unnest(ARRAY[1,2,3]):: int` or you can query by all model ids from a given model group or by dates,
it's up to you (as long as it returns a column `model_id`)

5. A list of dates query. Very similar to the previous point, you can either unnest a pre-defined list of dates
or execute a more complex query that returns a column `as_of_date`.

6. The models_dates_join_query is supposed to be fixed. Just change this if you are really sure. 
This query is necessary because different model_ids can be used for predicting at multiple as_of_dates we need to make sure
that model_id, as_of_date pairs really exist in a table containing predictions. 

7. The features query is used to specify a feature table or (joins of multiple feature tables)
that should be joined with the models_dates_join_query results.

8. Finally, the predictions query should return a model_id, as_of_date, entity_id, score, label_value, rank_abs and rank_pct columns.


### CLI

`triage crosstabs example/config/postmodeling_crosstabs.yaml` will run crosstabs for the given config YAML. The config YAML is highly dependent on what model ids and as-of-dates and feature tables are in the database. That example file needs to be modified to work with your experiment and postmodeling interests. Consult the instructions above for help in modifying the file.

### Python

This can be run using the `triage.component.postmodeling.crosstabs.run_crosstabs` function, which takes in a database engine and a loaded CrosstabsConfigLoader object. Example:

```python
from triage.component.postmodeling.crosstabs import CrosstabsConfigLoader, run_crosstabs
from sqlalchemy import create_engine

db_engine = create_engine(<mydburl>)
config = CrosstabsConfigLoader(config_file='example/config/postmodeling_crosstabs.yaml')
run_crosstabs(db_engine, config)
```
