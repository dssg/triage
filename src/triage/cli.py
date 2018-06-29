#!/usr/bin/env python
import yaml
import argparse
import os
from descriptors import cachedproperty
from argcmdr import RootCommand, Command, main, cmdmethod
from sqlalchemy.engine.url import URL
from triage.experiments import CONFIG_VERSION
from triage.component.audition import AuditionRunner
from triage.util.db import create_engine


class Triage(RootCommand):
    """manage Triage database and experiments"""

    def __init__(self, parser):
        parser.add_argument(
            '-d', '--dbfile',
            default='database.yaml',
            type=argparse.FileType('r'),
            help="database connection file",
        )

    @cachedproperty
    def db_url(self):
        dbconfig = yaml.load(self.args.dbfile)
        db_url = URL(
                'postgres',
                host=dbconfig['host'],
                username=dbconfig['user'],
                database=dbconfig['db'],
                password=dbconfig['pass'],
                port=dbconfig['port'],
            )
        return db_url


@Triage.register
class ConfigVersion(Command):
    """Return the experiment config version compatible with this installation of Triage"""
    def __call__(self, args):
        print(CONFIG_VERSION)


@Triage.register
class Audition(Command):
    """ Audition command to run or validate the config file
    """
    def __init__(self, parser):
        parser.add_argument(
            '-c', '--config',
            type=argparse.FileType('r'),
            default='audition_config.yaml',
            help="config file for audition",
        )
        parser.set_defaults(
            directory=None,
            validate=True,
        )

    @cachedproperty
    def runner(self):
        db_url = self.root.db_url
        dir_plot = self.args.directory
        config = yaml.load(self.args.config)
        db_engine = create_engine(db_url)
        return AuditionRunner(config, db_engine, dir_plot)

    def __call__(self, args):
        self['run'](args)

    @cmdmethod('-d', '--directory', default=None, help="directory to store the result plots from audition")
    @cmdmethod('-v', '--validate', action='store_true', help="validate before running audition")
    @cmdmethod('--no-validate', action='store_false', dest='validate', help="run audtion without validation")
    @cmdmethod('--validate-only', action='store_true', help="only validate the config file not running audition")
    def run(self, args):
        if args.validate_only:
            try:
                self.runner.validate()
            except Exception as err:
                raise(err)
        elif args.validate:
            try:
                self.runner.validate()
                self.runner.run()
            except Exception as err:
                raise(err)
        else:
            self.runner.run()

def execute():
    main(Triage)


if __name__ == '__main__':
    main(Triage)
