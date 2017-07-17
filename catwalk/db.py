import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import QueuePool

from results_schema import *


def ensure_db(engine):
    Base.metadata.create_all(engine)


def connect(poolclass=QueuePool):
    with open('database.yaml') as f:
        profile = yaml.load(f)
        dbconfig = {
            'host': profile['host'],
            'username': profile['user'],
            'database': profile['db'],
            'password': profile['pass'],
            'port': profile['port'],
        }
        dburl = URL('postgres', **dbconfig)
        return create_engine(dburl, poolclass=poolclass)
