import os.path

from alembic.config import Config
from alembic import command
import yaml
from sqlalchemy.engine.url import URL


from .schema import (
    Base,
    Experiment,
    FeatureImportance,
    IndividualImportance,
    ListPrediction,
    ExperimentMatrix,
    Matrix,
    ExperimentModel,
    ExperimentRun,
    ExperimentRunStatus,
    Model,
    ModelGroup,
    Subset,
    TestEvaluation,
    TrainEvaluation,
    TestPrediction,
    TrainPrediction,
    TestPredictionMetadata,
    TrainPredictionMetadata,
)


__all__ = (
    "Base",
    "Experiment",
    "FeatureImportance",
    "IndividualImportance",
    "ListPrediction",
    "ExperimentMatrix",
    "Matrix",
    "ExperimentModel",
    "ExperimentRun",
    "ExperimentRunStatus",
    "Model",
    "ModelGroup",
    "Subset",
    "TestEvaluation",
    "TrainEvaluation",
    "TestPrediction",
    "TrainPrediction",
    "TestPredictionMetadata",
    "TrainPredictionMetadata",
    "mark_db_as_upgraded",
    "upgrade_db",
)


def _base_alembic_args(db_config_filename=None):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, "alembic.ini")
    base = ["-c", alembic_ini_path]
    if db_config_filename:
        base += ["-x", "db_config_file={}".format(db_config_filename)]

    return base


def upgrade_db(db_engine=None, dburl=None):
    if db_engine:
        command.upgrade(alembic_config(dburl=db_engine.url), "head")
    elif dburl:
        command.upgrade(alembic_config(dburl=dburl), "head")
    else:
        raise ValueError("Must pass either a db_config_filehandle or a db_engine or a dburl")


REVISION_MAPPING = {
    "v1": "72ac5cbdca05",  # may add some logic to tag earlier ones if needed, could be 26 or 0d
    "v2": "72ac5cbdca05",
    "v3": "7d57d1cf3429",
    "v4": "89a8ce240bae",
    "v5": "2446a931de7a",
    "pre-v1": "8b3f167d0418",
}


def stamp_db(revision, dburl):
    command.stamp(alembic_config(dburl=dburl), revision)


def alembic_config(dburl):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, "alembic.ini")
    config = Config(alembic_ini_path)
    alembic_path = os.path.join(dir_path, "alembic")
    config.set_main_option("script_location", alembic_path)
    config.attributes["url"] = dburl
    return config
