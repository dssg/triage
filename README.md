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

6. If everything looks good, create a pull request!
