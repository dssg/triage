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


@Development.register
class Docs(Local):

    @local
    def prepare_dirtyduck(context):
        """convert dirtyduck org files to markdown format"""
        return context.local['emacs'][
            '--batch',
            '-l', 'org/publish.el', 'org/index.org',
            '--eval', '(org-publish "dirtyduck" t)',
        ]

    prepare_dirtyduck.__name__ = 'prepare-dirtyduck'  # or whatever; I just find underscores weird in sh
