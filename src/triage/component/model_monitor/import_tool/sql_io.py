import os
import yaml
import sqlalchemy as sa
from sqlalchemy.pool import NullPool


def get_db_config(fpath=None):
    if not fpath:
        fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                             "db_config.yaml")
    with open(fpath, mode='r') as f:
        return yaml.load(f)


def get_sqlalchemy_engine(db_config=None):
    return sa.create_engine('postgresql://{user}:{pwd}@{host}:{port}/{db}'.format(
        user=db_config["PG_USER"],
        pwd=db_config["PG_PASSWORD"],
        host=db_config["PG_HOST"],
        port=db_config["PG_PORT"],
        db=db_config["PG_DATABASE"]),
        poolclass=NullPool,
        connect_args={'options': '-csearch_path={}'.format(db_config["PG_SCHEMA_SEARCH_PATH"])})
