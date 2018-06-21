# results-schema
Store results of modeling runs in a relational database

## Quick Start

1. Install Triage

`pip install git+https://github.com/dssg/triage.git`

2. You can do the initial schema and table creation a couple of different ways, with a couple of different options for passing database credentials.

	- *Triage cli script*. Assuming you have triage installed from pip: `triage upgrade_db -d dbcredentials.yaml`, or just `triage upgrade_db` if you have a database credentials file at `database.yaml` (the default)
	- *From Python code*. If you are in a Python console or notebook, you can call `upgrade_db` with either sqlalchemy engine pointing to your database, or a filename similar to what's used to launch the cli script.

	```
	>>> from triage.component.results_schema import upgrade_db
	>>> upgrade_db(engine)
	>>> upgrade_db('database.yaml')
	```

This command will create a 'results' schema and the necessary tables.


## Modifying the schema

To make modifications to the schema, you should be working in a cloned version of the repository.

[Alembic](http://alembic.zzzcomputing.com/en/latest/tutorial.html) is a schema migrations library written in Python. It allows us to auto-generate migrations to run incremental database schema changes, such as adding or removing a column. This is done by comparing the definition of a schema in code with that of a live database. There are many valid ways to create migrations, which you can read about in [Alembic's documentation](http://alembic.zzzcomputing.com/en/latest/tutorial.html). But here is a common workflow we will use to modify the schema. Throughout, we'll use a wrapper script, `manage alembic`, bundled in the triage repository that wraps calls to the 'alembic' command with the correct options for running within triage. We'll only have a few examples here, but this script just passes all arguments to the 'alembic' command so if you know your way around alembic you can perform whatever operations you want there.

1. Create a candidate database for comparison if you don't have one already. You can use a toy database for this, or use your project database if the results schema has not been manually modified. Populate `database.yaml` in the repo root with the credentials, and upgrade it to the current HEAD: `manage alembic upgrade head`

2. Make the desired modifications to [results_schema.schema](src/triage/component/results_schema/schema.py).

3. From within the results schema directory, autogenerate a migration: `manage alembic revision --autogenerate` - This will look at the difference between your schema definition and the database, and generate a new file in results_schema/alembic/versions/.

4. Inspect the file generated in step 3 and make sure that the changes it is suggesting make sense. Make any modifications you want; the autogenerate functionality is just meant as a guideline.

5. Upgrade the database: `manage alembic upgrade head`

6. Update the [factories file](src/tests/results_tests/factories.py) with your changes - see more on factories below if you are unfamiliar with them.

7. If everything looks good, create a pull request!


## Using Factories

When you want to create rows of these results tables for a unit test, you can use the included factories to make this easier and with less boilerplate.  Factories allow you to only specify the attribute that are important to your test, and choose reasonable defaults for all other attributes. results_schema uses [FactoryBoy](http://factoryboy.readthedocs.io/en/latest/index.html) to accomplish this.

A simple example is to just instantiate an `EvaluationFactory`. `Evaluations` depend on `Models`, which depend on both `ModelGroups` and `Experiments`. So instantiating an `EvaluationFactory` actually creates four objects in the database.

```
from tests.results_tests.factories import EvaluationFactory, session

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
