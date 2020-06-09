import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import QueuePool

from triage.component.results_schema import Base


def ensure_db(engine):
    Base.metadata.create_all(engine)


def connect(poolclass=QueuePool):
    with open("database.yaml") as fd:
        config = yaml.full_load(fd)
        dburl = URL(
            "postgres",
            host=config["host"],
            username=config["user"],
            database=config["db"],
            password=config["pass"],
            port=config["port"],
        )
        return create_engine(dburl, poolclass=poolclass)
