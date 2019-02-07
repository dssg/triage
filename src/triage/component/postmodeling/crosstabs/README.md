# Crosstabs
 
The model crosstabs populates a table in a postgres database containing `model_id`, `as_of_date`, `threshold_unit`, `threshold_value`, and `feature_column`, such as the mean and standard deviation
of the given feature for the predicted-high-risk and predicted-low-risk group. In other words, this table provides simple
descriptives of the feature distributions for the high-risk and low-risk entities, given a specific model and decision
threshold.


# How to run

This module requires **Python 3.4+** so make sure to use the correct python version when running crosstabs.py.

Example of how to run crosstabs.py:

`python3 crosstabs.py --db sfpd_db_credentials.yaml --conf example_queries_sfpd.yaml`


crosstabs.py expects two arguments: `--db` and `--conf` which correspond to filepaths of a database credentials yaml file
and a crosstabs configuration file.

Example of a `your_db_credentials.yaml` file:

```
host: your_host
database: your_db 
user: alfonso
password: abcde
port: 5432
```
You can find an example of a crosstabs configuration in the `example_queries_sfpd.yaml` that is part of this repo. 
 
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

# Default Functions
`crosstabs.populate_crosstabs_table()` accepts a dictionary of `function_names`: `functions`, which are used
to populate the table. This functions simply receive two dataframes (one with high-risk entities' features,
one with low-risk entities' features) and return one or several columns of statistics. You can, of course,
add additional functions; the module by default provides `count_predicted_positive` (`_negative`), 
which simply gives the number of high (low) risk entities. This is independent of feature. It also provides
`mean_predicated_positive` (`_negative`), the feature's mean for the high (low) risk group;
`std_predicted_positive` (`_negative`), the feature's standard deviation for the high (low) risk group;
`ratio_predicted_positive_over_predicted_negative`, the high-risk group's feature mean divided by the 
low-risk group's feature mean;
`ttest_T` and `ttest_p`, the corresponding T-statistic and p-value.
