#!/usr/bin/env python

from argcmdr import RootCommand, Command, main
from triage.experiments import CONFIG_VERSION


class Triage(RootCommand):
    """manage Triage database and experiments"""

    def __init__(self, parser):
        parser.add_argument(
            '-d', '--dbfile',
            default='database.yaml',
            help="database connection file",
        )


@Triage.register
class ConfigVersion(Command):
    """Return the experiment config version compatible with this installation of Triage"""
    def __call__(self, args):
        print(CONFIG_VERSION)


def execute():
    main(Triage)


if __name__ == '__main__':
    main(Triage)
