from __future__ import with_statement

import os

import yaml
import re
from alembic import context
from sqlalchemy import create_engine
from sqlalchemy import pool
from sqlalchemy.engine.url import URL

from triage.component.results_schema import Base


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_excludes_from_config(config_, type_="tables"):
    excludes = config_.get(type_, None)
    if excludes is not None:
        excludes = excludes.split(",")
    excludes = excludes or []
    return excludes


excluded_tables = get_excludes_from_config(config.get_section('exclude'), "tables")
excluded_indices = get_excludes_from_config(config.get_section('exclude'), "indices")


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table":
        for table_pat in excluded_tables:
            if re.match(table_pat, name):
                return False
        return True

    elif type_ == "index":
        for index_pat in excluded_indices:
            if re.match(index_pat, name):
                return False
        return True

    else:
        return True


url = None

if "url" in config.attributes:
    url = config.attributes["url"]

if not url:
    url = os.environ.get("DBURL", None)

if not url:
    db_config_file = context.get_x_argument("db_config_file").get(
        "db_config_file", None
    )
    if not db_config_file:
        raise ValueError("No database connection information found")

    with open(db_config_file) as fd:
        config = yaml.full_load(fd)
        url = URL(
            "postgres",
            host=config["host"],
            username=config["user"],
            database=config["db"],
            password=config["pass"],
            port=config["port"],
        )

def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        version_table="results_schema_versions",
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table="results_schema_versions",
            include_schemas=True,
            include_object=include_object,
        )
        connection.execute('set search_path to "{}", public'.format("results"))

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
