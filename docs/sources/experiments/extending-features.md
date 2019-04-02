# Extending Feature Generation

This document describes how to extend Triage's feature generation capabilities by writing new FeatureBlock classes and incorporating them into Experiments.

## What is a FeatureBlock?

A FeatureBlock represents a single feature table in the database and how to generate it. If you're familiar with `collate` parlance, a `SpacetimeAggregation` is similar in scope to a FeatureBlock. A `FeatureBlock` class can be instantiated with whatever arguments it needs,and from there can provide queries to produce its output feature table. Full-size Triage experiments tend to contain multiple feature blocks. These all live in a collection as the `experiment.feature_blocks` property in the Experiment.

## What existing FeatureBlock classes can I use?

Class name | Experiment config key | Use
------------ | ------------- | ------------
triage.component.collate.SpacetimeAggregation | spacetime_aggregation  | Temporal aggregations of event-based data

## Writing a new FeatureBlock class

The `FeatureBlock` base class defines a set of abstract methods that any child class must implement, as well as a number of initialization arguments that it must take and implement in order to fulfill expectations Triage users have on feature generators. Triage expects these classes to define the queries they need to run, as opposed to generating the tables themselves, so that Triage can implement scaling by parallelization.

### Abstract methods

Any method here without parentheses afterwards is expected to be a property.

Method | Task | Return Type
------------ | ------------- | -------------
feature_columns | The list of feature columns in the final, postimputation table. Should exclude any index columns (e.g. entity id, date) | list
preinsert_queries | Return all queries that should be run before inserting any data. The creation of your feature table should happen here, and is expected to have `entity_id(integer)` and `as_of_date(timestamp)` columns. | list
insert_queries | Return all inserts to populate this data. Each query in this list should be parallelizable, and should be valid after all `preinsert_queries` are run. | list
postinsert_queries | Return all queries that should be run after inserting all data | list
imputation_queries | Return all queries that should be run to fill in missing data with imputed values. | list

Any of the query list properties can be empty: for instance, if your implementation doesn't have inserts separate from table creation and is just one big query (e.g. a `CREATE TABLE AS`), you could just define `preinsert_queries` so be that one mega-query and leave the other properties as empty lists.

### Properties Provided by Base Class

There are several attributes/properties that can be used within subclass implementations that the base class provides. Triage experiments take care of providing this data during runtime: if you want to instantiate a FeatureBlock object on your own, you'll have to provide them in the constructor.

Name | Type | Purpose
------------ | ------------- | -------------
as_of_dates | list | Features are created "as of" specific dates, and expects that each of these dates will be populated with a row for each member of the cohort on that date.
cohort_table | string | The final shape of the feature table should at least include every entity id/date pair in this cohort table.
final_feature_table_name | string | The name of the final table with all features filled in (no missing values). This is provided by the user in feature config, as the key that corresponds to the configuration block that instantiates.
db_engine | sqlalchemy.engine | The engine to use to access the database. Although these instances are mostly returning queries, the engine may be useful for implementing imputation.
features_schema_name | string | The database schema where all feature tables should reside. Defaults to None, which ends up in the public schema.
feature_start_time | string/datetime | A time before which no data should be considered for features. This is generally only applicable if your FeatureBlock is doing temporal aggregations. Defaults to None, which means no data will be excluded.
features_ignore_cohort | bool | If True (the default), features are only computed for members of the cohort. If False, the shape of the final feature table could include more.


`FeatureBlock` child classes can, and in almost all cases will, include more configuration at initialization time that are specific to them. They probably also define many more methods to use internally. But as long as they adhere to this interface, they'll work with Triage.

### Making the new FeatureBlock available to experiments

Triage Experiments run on serializable configuration, and although it's possible to take fully generated `FeatureBlock` instances and bypass this (e.g. `experiment.feature_blocks = <my_collection_of_feature_blocks>`), it's not recommended. The last step is to pick a config key for use within the `features` key of experiment configs, within `triage.component.architect.feature_block_generators.FEATURE_BLOCK_GENERATOR_LOOKUP` and point it to a function that instantiates a bunch of your objects based on config.

## Example

That's a lot of information! Let's see this in action. Let's say that we want to create a very flexible type of feature that simply runs a configured query with a parametrized as-of-date and returns its result as a feature.

```python
from triage.component.architect.feature_block import FeatureBlock


class SimpleQueryFeature(FeatureBlock):
    def __init__(self, query, *args, **kwargs):
        self.query = query
        super().__init__(*args, **kwargs)

    @property
    def final_feature_table_name(self):
        return f"{self.features_schema_name}.mytable"

    @property
    def feature_columns(self):
        return ['myfeature']

    @property
    def preinsert_queries(self):
        return [f"create table {self.final_feature_table_name}" "(entity_id bigint, as_of_date timestamp, myfeature float)"]

    @property
    def insert_queries(self):
        if self.features_ignore_cohort:
            final_query = self.query
        else:
            final_query = f"""
                select * from (self.query) raw
                join {self.cohort_table} using (entity_id, as_of_date)
            """
        return [
            final_query.format(as_of_date=date)
            for date in self.as_of_dates
        ]

    @property
    def postinsert_queries(self):
        return [f"create index on {self.final_feature_table_name} (entity_id, as_of_date)"]

    @property
    def imputation_queries(self):
        return [f"update {self.final_feature_table_name} set myfeature = 0.0 where myfeature is null"]
```

This class would allow many different uses: basically any query a user can come up with would be a feature. To instantiate this class outside of triage with a simple query, you could:

```python
feature_block = SimpleQueryFeature(
    query="select entity_id, as_of_date, quantity from source_table where date < '{as_of_date}'",
    as_of_dates=["2016-01-01"],
    cohort_table="my_cohort_table",
    db_engine=triage.create_engine(<..mydbinfo..>)
)

feature_block.run_preimputation()
feature_block.run_imputation()
```

To use it from a Triage experiment, modify `triage.component.architect.feature_block_generators.py` and submit a pull request:

Before:

```python
FEATURE_BLOCK_GENERATOR_LOOKUP = {
    'spacetime_aggregations': generate_spacetime_aggregations
}
```

After:

```python
FEATURE_BLOCK_GENERATOR_LOOKUP = {
    'spacetime_aggregations': generate_spacetime_aggregations,
    'simple_query': SimpleQueryFeature,
}
```

At this point, you could use it in an experiment configuration like this:

```yaml

features:
    simple_query:
        - query: "select entity_id, as_of_date, quantity from source_table where date < '{as_of_date}'"
        - query: "select entity_id, as_of_date, other_quantity from other_source_table where date < '{as_of_date}'"
```
