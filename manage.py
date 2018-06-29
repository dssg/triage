import argparse
from pathlib import Path

from argcmdr import local, LocalRoot


ROOT_PATH = Path(__file__).parent.resolve()


class Development(LocalRoot):
    """Commands to aid in Triage library development"""
    pass


@Development.register
@local('remainder', metavar='alembic arguments', nargs=argparse.REMAINDER)
def alembic(context, args):
    """Configuration wrapper to use the Alembic schema migrations library for Triage development"""
    return context.local['env'][
        'PYTHONPATH=' + str(ROOT_PATH / 'src'),
        'alembic',
        '-c', ROOT_PATH / 'src' / 'triage' / 'component' / 'results_schema' / 'alembic.ini',
        '-x', 'db_config_file=database.yaml',
        args.remainder,
    ]
