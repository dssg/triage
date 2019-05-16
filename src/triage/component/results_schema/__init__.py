import os.path

import logging
from alembic.config import Config
from alembic import script
from alembic import command
from triage import create_engine
from triage.database_reflection import table_exists


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


def upgrade_if_clean(dburl):
    """Upgrade the database only if the results schema hasn't been created yet.

    Raises: ValueError if the database results schema version does not equal the code's version
    """
    alembic_cfg = alembic_config(dburl)
    engine = create_engine(dburl)
    script_ = script.ScriptDirectory.from_config(alembic_cfg)
    if not table_exists('results_schema_versions', engine):
        logging.info("No results_schema_versions table exists, which means that this installation "
                     "is fresh. Upgrading db.")
        upgrade_db(dburl=dburl)
        return
    with engine.begin() as conn:
        current_revision = conn.execute('select version_num from results_schema_versions limit 1').scalar()
        logging.info("Database's results schema version is %s", current_revision)
        triage_head = script_.get_current_head()
        logging.info("Code's results schema version is %s", triage_head)
        database_is_ahead = not any(migration.revision == current_revision for migration in script_.walk_revisions())
        if database_is_ahead:
            raise ValueError(
                "Your database's results schema version, %s, is not a known revision to this"
                "version of Triage knows about. Usually, this happens if you use a branch with a"
                "new results schema version and upgrade the database to that version. To use this"
                "version of Triage, you will likely need to check out that branch out "
                "and downgrade to %s",
                current_revision,
                triage_head
            )
        elif current_revision != triage_head:
            raise ValueError(
                "Your database's results schema revision, %s, does not equal the revision used"
                "by this version of Triage. However, your database can be upgraded to this "
                "revision. If you would like to upgrade the database, call "
                "`triage db upgrade` if this is installed version of Triage, or "
                "`manage alembic upgrade head` if this is a cloned Triage repository."
                "The database changes may take a long time if this is a heavily populated database"
                "Otherwise, you can also downgrade your Triage version to match your database."
            )


def alembic_config(dburl):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, "alembic.ini")
    config = Config(alembic_ini_path)
    alembic_path = os.path.join(dir_path, "alembic")
    config.set_main_option("script_location", alembic_path)
    config.attributes["url"] = dburl
    return config
