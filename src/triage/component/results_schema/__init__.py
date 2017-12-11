import alembic.config
from .schema import *
import os


def _base_alembic_args(db_config_filename=None):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, 'alembic.ini')
    base = ['-c', alembic_ini_path]
    if db_config_filename:
        base += ['-x', 'db_config_file={}'.format(db_config_filename)]

    return base


def upgrade_db(db_config_filename=None):
    args = _base_alembic_args(db_config_filename) + ['--raiseerr', 'upgrade', 'head']
    alembic.config.main(argv=args)


def mark_db_as_upgraded(db_config_filename=None):
    args = _base_alembic_args(db_config_filename) + ['--raiseerr', 'stamp', 'head']
    alembic.config.main(argv=args)
