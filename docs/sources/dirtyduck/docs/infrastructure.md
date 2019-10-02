# Infrastructure

In every data science project you will need several tools to help analyze the data in an efficient<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> manner. Examples include a place to store the data (a database management system or **DBMS**); a way to put your model to work, i.e. a way that allows the model to ingest new data and make predictions (an **API**); and a way to examine the performance of trained models (monitor tools).

This tutorial includes a script for managing the infrastructure<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> in a transparent way.

The infrastructure of this tutorial has *four* pieces:

-   a `postgresql` database called `food_db`,
-   a container that executes `triage` experiments (we will use this when trying to scale up),
-   a container for interacting with the data called `bastion`.

`bastion` includes a `postgresql` client (so you can interact with the database)<sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup> and a full `python` environment (so you can code or modify the things for the tutorial).

The only thing you need installed on your laptop is `docker`.

From your command line (terminal) run the following from the repo directory:

```shell
    ./tutorial.sh
```

```org
   Usage: ./tutorial.sh {start|stop|build|rebuild|run|logs|status|destroy|all|}

   OPTIONS:
      -h|help             Show this message
      start
      stop
      rebuild
      status
      destroy
      -t|triage
      -a|all

   INFRASTRUCTURE:
      Build the infrastructure:
           $ ./tutorial.sh start

      Check the status of the containers:
           $ ./tutorial.sh status

      Stop the tutorial's infrastructure:
           $ ./tutorial.sh stop

      Destroy all the resources related to the tutorial:
           $ ./tutorial.sh destroy

      View the infrastructure logs:
           $ ./tutorial.sh -l

   EXPERIMENTS:
      NOTE:
         The following commands assume that "sample_experiment_config.yaml"
         is located inside the triage/experiments directory

      Run one experiment:
           $ ./tutorial.sh -t --config_file sample_experiment_config.yaml run

      Run one experiment, do not replace existing matrices or models, and enable debug:
           $ ./tutorial.sh -t --config_file sample_experiment_config.yaml --no-replace --debug run

      Validate experiment configuration file:
           $ ./tutorial.sh triage --config_file sample_experiment_config.yaml validate

      Show the experiment's temporal cross-validation blocks:
           $ ./tutorial.sh -t --config_file sample_experiment_config.yaml show-temporal-blocks

      Plot model number 4 (for Decision Trees and Random Forests):
           $ ./tutorial.sh -t --config_file sample_experiment_config.yaml show_model_plot --model 4

      Triage help:
           $ ./tutorial.sh triage --help

```

Following the instructions on the screen, we can start the infrastructure with:

```sh
    ./tutorial.sh start
```

You can check that everything is running smoothly with `status`

```sh
    ./tutorial.sh status
```

```org
    Name                Command              State           Ports
   ------------------------------------------------------------------------
   food_db   docker-entrypoint.sh postgres   Up      0.0.0.0:5434->5432/tcp
```

To access `bastion`, where the `postgresql` client is, type:

```sh
   ./tutorial.sh bastion
```

Your prompt should change to something like:

    root@485373fb3c64:/$

**NOTE**: The number you see will be different (i.e. not `485373fb3c64`).

Inside `bastion`, type the next command to connect to the database

```sh
   psql ${DATABASE_URL}
```

The prompt will change again to (or something *very* similar):

    psql (9.6.7, server 10.2 (Debian 10.2-1.pgdg90+1))
    WARNING: psql major version 9.6, server major version 10.
          Some psql features might not work.
    Type "help" for help.

    food=#

The previous command is using `psql`, a powerful command line client for the Postgresql database. If you want to use this client fully, check [psql's documentation](https://www.postgresql.org/docs/10/static/app-psql.html).

The database is running and it's named `food`. It should contain a single table named `inspections` in the `schema` `raw`. Let's check the structure of the `inspections` table. Type the following command:

```sql
    \d raw.inspections
```

| Table "raw.inspections" |                   |           |          |         |
|----------------------- |----------------- |--------- |-------- |------- |
| Column                  | Type              | Collation | Nullable | Default |
| inspection              | character varying |           | not null |         |
| dba<sub>name</sub>      | character varying |           |          |         |
| aka<sub>name</sub>      | character varying |           |          |         |
| license<sub>num</sub>   | numeric           |           |          |         |
| facility<sub>type</sub> | character varying |           |          |         |
| risk                    | character varying |           |          |         |
| address                 | character varying |           |          |         |
| city                    | character varying |           |          |         |
| state                   | character varying |           |          |         |
| zip                     | character varying |           |          |         |
| date                    | date              |           |          |         |
| type                    | character varying |           |          |         |
| results                 | character varying |           |          |         |
| violations              | character varying |           |          |         |
| latitude                | numeric           |           |          |         |
| longitude               | numeric           |           |          |         |
| location                | character varying |           |          |         |

That's it. We will work from this table of raw data.

You can disconnect from the database typing `\q`. But don't leave the database yet! We still need to do a lot of things <sup><a id="fnr.4" class="footref" href="#fn.4">4</a></sup>

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> Reproducible, scalable, flexible, etc.

<sup><a id="fn.2" class="footnum" href="#fnr.2">2</a></sup> And other things through this tutorial, like the execution of the model training, etc.

<sup><a id="fn.3" class="footnum" href="#fnr.3">3</a></sup> If you have a postgresql client installed, you can use `psql -h 0.0.0.0 -p 5434 -d food -U food_user` rather than the `bastion` container.

<sup><a id="fn.4" class="footnum" href="#fnr.4">4</a></sup> Welcome to the not-so-sexy part of the (supposedly) *sexiest job* of the XXI century.
