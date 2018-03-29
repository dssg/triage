# Crosstabs Tables
By default, these modules create and populate the `results.crosstabs` table. This table lists simple statistics per 
`model_id`, `as_of_date`, `threshold_unit`, `threshold_value`, and `feature_column`, such as the mean and standard deviation
of the given feature for the predicted-high-risk and predicted-low-risk group. In other words, this table provides simple
descriptives of the feature distributions for the high-risk and low-risk entities, given a specific model and decision
threshold.

# Creating Tables
Make sure that this repo's `database_credentials.json` is populated with your Postgres DB credentials. 
`crosstabs.create_crosstabs_tabs()` will set up the empty `results.crosstabs` table, after dropping the table if 
it already exists.

# Populating the Table
`crosstabs.populate_crosstabs_table` will populate the above-mentioned table. Check its docstring for details; you
will need to provide a list of `(model_id, as_of_date)` tuples for which you want to run the 
crosstabs. You will also need to implement a function similar to `crosstabs.get_features_query`.
This function needs to return a query which, when run against the Postgres DB, returns a 
table with feature columns information per entity. See 
`crosstabs.get_sfpd_features_query for an example implementation for the
police project's staging schema. Alternatively, you can supply the features table as 
a `Pandas.DataFrame`; again, check the `crosstabs.populate_crosstabs_table` docstring 
for details.

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
