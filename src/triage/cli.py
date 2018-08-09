#!/usr/bin/env python
import yaml
import argparse
from descriptors import cachedproperty
from argcmdr import RootCommand, Command, main, cmdmethod
from sqlalchemy.engine.url import URL

from triage.component.architect.feature_generators import FeatureGenerator
from triage.component.audition import AuditionRunner
from triage.component.results_schema import upgrade_db, stamp_db, REVISION_MAPPING
from triage.experiments import CONFIG_VERSION
from triage.util.db import create_engine

import logging

logging.basicConfig(level=logging.INFO)


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
class Experiment(Command):
    """Validate and run experiments, manage experiment database"""
    @cmdmethod
    def upgradedb(self, args):
        """Upgrade triage results database"""
        upgrade_db(args.dbfile)

    @cmdmethod('configversion', choices=REVISION_MAPPING.keys(), help='config version of last experiment you ran')
    def stampdb(self, args):
        """Instruct the triage results database to mark itself as updated to a known version without doing any upgrading.
        
        Use this if the database was created without an 'alembic_version' table. Uses the config version of your experiment to infer what database version is suitable.
        """
        revision = REVISION_MAPPING[args.configversion]
        print(f"Based on config version {args.configversion} "
              f"we think your results schema is version {revision} and are upgrading to it")
        stamp_db(revision, args.dbfile)

    @cmdmethod
    def configversion(self, args):
        """Return the experiment config version compatible with this installation of Triage"""
        print(CONFIG_VERSION)

    @cmdmethod('as_of_date', help='The date as of which to run features')
    @cmdmethod('feature_config_file', type=argparse.FileType('r'), help='Feature config YAML file, containing a list of feature_aggregation objects')
    def featuretest(self, args):
        """Run a feature aggregation for an as-of-date"""
        db_engine = create_engine(self.root.db_url)
        feature_config = yaml.load(args.feature_config_file)

        FeatureGenerator(db_engine, 'features_test').create_features_before_imputation(
            feature_aggregation_config=feature_config,
            feature_dates=[args.as_of_date]
        )
        logging.info('Features created for feature_config %s and date %s', feature_config, args.as_of_date)


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
