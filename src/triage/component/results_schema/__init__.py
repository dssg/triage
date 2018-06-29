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
    Matrix,
    Model,
    ModelGroup,
    TestEvaluation,
    TrainEvaluation,
    TestPrediction,
    TrainPrediction
)


__all__ = (
    'Base',
    'Experiment',
    'FeatureImportance',
    'IndividualImportance',
    'ListPrediction',
    'Matrix',
    'Model',
    'ModelGroup',
    'TestEvaluation',
    'TrainEvaluation',
    'TestPrediction',
    'TrainPrediction',
    'mark_db_as_upgraded',
    'upgrade_db',
)


def _base_alembic_args(db_config_filename=None):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, 'alembic.ini')
    base = ['-c', alembic_ini_path]
    if db_config_filename:
        base += ['-x', 'db_config_file={}'.format(db_config_filename)]

    return base


def upgrade_db(db_config_filename=None, db_engine=None):
    if db_config_filename:
        command.upgrade(alembic_config(db_config_filename=db_config_filename), 'head')
    elif db_engine:
        command.upgrade(alembic_config(dburl=db_engine.url), 'head')
    else:
        raise ValueError('Must pass either a db_config_filename or a db_engine')


REVISION_MAPPING = {
    'v1': '72ac5cbdca05',  # may add some logic to tag earlier ones if needed, could be 26 or 0d
    'v2': '72ac5cbdca05',
    'v3': '7d57d1cf3429',
    'v4': '89a8ce240bae',
    'v5': '2446a931de7a',
    'pre-v1': '8b3f167d0418'
}


def stamp_db(revision, db_config_filename):
    command.stamp(alembic_config(db_config_filename=db_config_filename), revision)


def alembic_config(db_config_filename=None, dburl=None):
    if db_config_filename:
        with open(db_config_filename) as f:
            config = yaml.load(f)
            dburl = URL(
                'postgres',
                host=config['host'],
                username=config['user'],
                database=config['db'],
                password=config['pass'],
                port=config['port'],
            )
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, 'alembic.ini')
    config = Config(alembic_ini_path)
    alembic_path = os.path.join(dir_path, 'alembic')
    config.set_main_option('script_location', alembic_path)
    config.attributes['url'] = dburl
    return config
