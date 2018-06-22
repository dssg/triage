from __future__ import with_statement

import os
from alembic import context
from sqlalchemy import pool
from logging.config import fileConfig
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine

from triage.component.model_monitor.mm_results_schema import Base
from triage.component.model_monitor.import_tool.sql_io import get_db_config, get_sqlalchemy_engine

config = context.config

fileConfig(config.config_file_name)

target_metadata = Base.metadata

url = os.environ.get("DBURL", None)

if not url:
    db_config_file = (context.get_x_argument('db_config_file')
                      .get('db_config_file', None))
    if not db_config_file:
        raise ValueError('No database connection information found')

    db_config = get_db_config(db_config_file)
    url = URL(
        'postgres',
        host=db_config['PG_HOST'],
        port=db_config['PG_PORT'],
        database=db_config['PG_DATABASE'],
        username=db_config['PG_USER'],
        password=db_config['PG_PASSWORD'],
    )


def run_migrations_offline():
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        version_table='results_schema_versions'
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = create_engine(
        url,
        poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table='results_schema_versions',
            include_schemas=True,
        )
        connection.execute('set search_path to "{}", public'.format('model_monitor'))

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
