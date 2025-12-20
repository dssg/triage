# Setting up the Infrastructure

In every data science project you will need several tools to help
analyze the data in an efficient[^1] manner. Examples include a place
to store the data (e.g. database management system or **DBMS**); a way to
put your model to work, e.g. a way that allows the model to ingest new
data and make predictions (an **API**); and a way to examine the
performance of trained models (e.g. monitor tools).

This tutorial uses [just](https://github.com/casey/just) for managing the infrastructure[^2] in
a transparent way.

The infrastructure of this tutorial has *three* pieces:

-   a `postgresql` database called `food_db`,
-   a container that executes `triage` experiments (we will use this when trying to scale up),
-   a container for interacting with the data called `bastion`.

`bastion` includes a `postgresql` client (so you can interact with the
database)[^3] and a full `python` environment (so you can code or
modify the things for the tutorial).

The only thing you need installed on your laptop is `docker` and `just`.

From your command line (terminal) run the following command from the repo directory:

```shell
    just --list
```

```org
Available DirtyDuck tutorial commands:

   tutorial-up          Start DirtyDuck tutorial database
   tutorial-down        Stop DirtyDuck tutorial
   tutorial-shell       Launch DirtyDuck bastion shell
   tutorial-build       Build DirtyDuck images
   tutorial-rebuild     Rebuild DirtyDuck images (no cache)
   tutorial-status      Show DirtyDuck container status
   tutorial-logs        View DirtyDuck logs
   tutorial-clean       Clean up DirtyDuck resources (removes containers, images, volumes)

INFRASTRUCTURE:
   Build the DB's infrastructure:
        $ just tutorial-up

   Check the status of the containers:
        $ just tutorial-status

   Stop the tutorial's DB's infrastructure:
        $ just tutorial-down

   Destroy all the resources related to the tutorial:
        $ just tutorial-clean

   View the infrastructure logs:
        $ just tutorial-logs

```

Following the instructions on the screen, we can start the infrastructure with:

```sh
    just tutorial-up
```

You can check that everything is running smoothly with `status` by
using the following command:

```sh
    just tutorial-status
```

```org
    Name                Command              State           Ports
   ------------------------------------------------------------------------
   tutorial_db   docker-entrypoint.sh postgres   Up      0.0.0.0:5434->5432/tcp
```

To access `bastion`, where the `postgresql` client is, submit the command:

```sh
   just tutorial-shell
```

Your prompt should change to something like:

    [triage@dirtyduck$:/dirtyduck]#

**NOTE**: The number you see will be different (i.e. not `485373fb3c64`).

Inside `bastion`, type the next command to connect to the database

```sh
   psql ${DATABASE_URL}
```

The prompt will change again to (or something *very* similar):

    psql (12.3 (Debian 12.3-1.pgdg100+1), server 12.2 (Debian 12.2-2.pgdg100+1))
    Type "help" for help.

    food=#

The previous command is using `psql`, a powerful command line client
for the Postgresql database. If you want to use this client fully,
check [psql's
documentation](https://www.postgresql.org/docs/10/static/app-psql.html).

The database is now running and is named `food`. It should contain a
single table named `inspections` in the `schema` `raw`. Let's check
the structure of the `inspections` table. Type the following command:

```sql
    \d raw.inspections
```

|      Column      |       Type        | Collation | Nullable | Default|
|------------------|-------------------|-----------|----------|---------|
| inspection       | character varying |           | not null |         |
| dba\_name         | character varying |           |          ||
| aka\_name         | character varying |           |          ||
| license\_num      | numeric           |           |          ||
| facility\_type    | character varying |           |          ||
| risk             | character varying |           |          ||
| address          | character varying |           |          ||
| city             | character varying |           |          ||
| state            | character varying |           |          ||
| zip              | character varying |           |          ||
| date             | date              |           |          ||
| type             | character varying |           |          ||
| results          | character varying |           |          ||
| violations       | character varying |           |          ||
| latitude         | numeric           |           |          ||
| longitude        | numeric           |           |          ||
| location         | character varying |           |          ||


That's it! We will work with this table of raw inspections data.

You can disconnect from the database by typing `\q`. But don't leave
the database yet! We still need to do a lot of things[^4]



[^1]: Reproducible, scalable, flexible, etc.

[^2]: And other things through this tutorial, like the execution of the model training, etc.

[^3]: If you have a postgresql client installed, you can use `psql -h 0.0.0.0 -p 5434 -d food -U food_user` rather than the `bastion` container.

[^4]: Welcome to the not-so-sexy part of the (supposedly) *sexiest job* of the XXI century.
