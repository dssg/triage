# results-schema
Store results of modeling runs in a relational database

## Quick Start

1. Install

`pip install git+https://github.com/dssg/results-schema.git`

2. Create a YAML file with your database credentials (see example_db_config.yaml), or an environment variable 'DBURL' with a connection string. The database must be created already.

3. Call 'upgrade_db' function from Python console or script

```
>>> from results_schema import upgrade_db
>>> upgrade_db('my_db_config.yaml')
```

This command will create a 'results' schema and the necessary tables.


## Modifying the schema

[Alembic](http://alembic.zzzcomputing.com/en/latest/tutorial.html) is a schema migrations library written in Python. It allows us to auto-generate migrations to run incremental database schema changes, such as adding or removing a column. This is done by comparing the definition of a schema in code with that of a live database. There are many valid ways to create migrations, which you can read about in [Alembic's documentation](http://alembic.zzzcomputing.com/en/latest/tutorial.html). But here is a common workflow we will use to modify the schema.

1. Have a candidate database for comparison. You can use a toy database for this that you upgrade to the current master, or use your project database if the results schema has not been manually modified.

2. Make the desired modifications to [results_schema.schema](results_schema/schema.py).

3. Autogenerate a migration: `alembic -c results_schema/alembic.ini -x db_config_file=my_db_config.yaml revision --autogenerate` - This will look at the difference between your schema definition and the database, and generate a new file in results_schema/alembic/versions/.

4. Inspect the file generated in step 3 and make sure that the changes it is suggesting make sense. Make any modifications you want; the autogenerate functionality is just meant as a guideline.

5. Upgrade the database: `alembic -c results_schema/alembic.ini -x db_config_file=my_db_config.yaml upgrade head` 

6. Update the [factories file](results_schema/factories/__init__.py) with your changes - see more on factories below if you are unfamiliar with them.

7. If everything looks good, create a pull request!


## Using Factories

When you want to create rows of these results tables for a unit test, you can use the included factories to make this easier and with less boilerplate.  Factories allow you to only specify the attribute that are important to your test, and choose reasonable defaults for all other attributes. results_schema uses [FactoryBoy](http://factoryboy.readthedocs.io/en/latest/index.html) to accomplish this.

A simple example is to just instantiate an `EvaluationFactory`. `Evaluations` depend on `Models`, which depend on both `ModelGroups` and `Experiments`. So instantiating an `EvaluationFactory` actually creates four objects in the database.

```
from results_schema.factories import EvaluationFactory, session

init_engine(engine)
EvaluationFactory()
session.commit()

results = engine.execute('select model_id, metric, parameter, value from results.evaluations')
for row in results:
	print(row)
```

```
(1, 'precision@', '100_abs', Decimal('0.76'))
```

This is all well and good, but often your tests will require some more control over the relationships between the objects you create, like creating different evaluations keyed to the same model. You do this by instantiating a `ModelFactory` first and then passing that to each `EvaluationFactory`:

```
init_engine(engine)

model = ModelFactory()
for metric, value in [
	('precision@', 0.4),
	('recall@', 0.3),
]:
	EvaluationFactory(
		model_rel=model,
		metric=metric,
		parameter='100_abs',
		value=value
	)
session.commit()
results = engine.execute('select model_id, metric, parameter, value from results.evaluations')
for row in results:
	print(row)
```

```
(1, 'precision@', '100_abs', Decimal('0.4'))
(1, 'recall@', '100_abs', Decimal('0.3'))
```
