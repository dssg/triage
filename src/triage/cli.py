#!/usr/bin/env python
import argparse
import importlib.util
import logging
import os.path
import yaml
from datetime import datetime

from descriptors import cachedproperty
from argcmdr import RootCommand, Command, main, cmdmethod
from sqlalchemy.engine.url import URL

from triage.component.architect.feature_generators import FeatureGenerator
from triage.component.architect.cohort_table_generators import CohortTableGenerator
from triage.component.audition import AuditionRunner
from triage.component.results_schema import upgrade_db, stamp_db, REVISION_MAPPING
from triage.component.timechop import Timechop
from triage.component.timechop.plotting import visualize_chops
from triage.component.catwalk.storage import Store
from triage.component.catwalk.storage import CSVMatrixStore, HDFMatrixStore
from triage.experiments import (
    CONFIG_VERSION,
    MultiCoreExperiment,
    SingleThreadedExperiment,
)
from triage.util.db import create_engine

logging.basicConfig(level=logging.INFO)


def natural_number(value):
    natural = int(value)
    if natural <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid natural number")
    return natural


def valid_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid date (format: YYYY-MM-DD)"
        )


class Triage(RootCommand):
    """manage Triage database and experiments"""

    SETUP_FILE_DEFAULT = os.path.abspath('experiment.py')

    def __init__(self, parser):
        parser.add_argument(
            "-d",
            "--dbfile",
            default="database.yaml",
            type=argparse.FileType("r"),
            help="database connection file",
        )
        parser.add_argument(
            '-s', '--setup',
            help=f"file path to Python module to import before running the "
                 f"Experiment (default: {self.SETUP_FILE_DEFAULT})",
        )

    def setup(self):
        if not self.args.setup and not os.path.exists(self.SETUP_FILE_DEFAULT):
            return

        setup_path = self.args.setup or self.SETUP_FILE_DEFAULT
        logging.info("Loading setup module at %s", setup_path)
        spec = importlib.util.spec_from_file_location("triage_config", setup_path)
        triage_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triage_config)
        logging.info(f"Setup module loaded")

    @cachedproperty
    def db_url(self):
        dbconfig = yaml.load(self.args.dbfile)
        db_url = URL(
            "postgres",
            host=dbconfig["host"],
            username=dbconfig["user"],
            database=dbconfig["db"],
            password=dbconfig["pass"],
            port=dbconfig["port"],
        )
        return db_url

    @cmdmethod
    def configversion(self, args):
        """Check the experiment config version compatible with this installation of Triage"""
        print(CONFIG_VERSION)

@Triage.register
class FeatureTest(Command):
    """Test a feature aggregation by running it for one date"""

    def __init__(self, parser):
        parser.add_argument(
            "feature_config_file",
            type=argparse.FileType("r"),
            help="Feature config YAML file, containing a list of feature_aggregation objects",
        )
        parser.add_argument(
            "as_of_date",
            type=valid_date,
            help="The date as of which to run features. Format YYYY-MM-DD",
        )

    def __call__(self, args):
        self.root.setup()  # Loading configuration (if exists)
        db_engine = create_engine(self.root.db_url)
        full_config = yaml.load(args.feature_config_file)
        feature_config = full_config['feature_aggregations']
        cohort_config = full_config.get('cohort_config', None)
        if cohort_config:
            CohortTableGenerator(
                cohort_table_name="features_test.test_cohort",
                db_engine=db_engine,
                query=cohort_config["query"],
                replace=True
            ).generate_cohort_table(as_of_dates=[args.as_of_date])

        FeatureGenerator(db_engine, "features_test").create_features_before_imputation(
            feature_aggregation_config=feature_config,
            feature_dates=[args.as_of_date],
            state_table="features_test.test_cohort"
        )
        logging.info(
            "Features created for feature_config %s and date %s",
            feature_config,
            args.as_of_date,
        )


@Triage.register
class Experiment(Command):
    """Run a full modeling experiment"""

    matrix_storage_map = {
        "csv": CSVMatrixStore,
        "hdf": HDFMatrixStore,
    }
    matrix_storage_default = "csv"

    def __init__(self, parser):
        parser.add_argument(
            "config",
            help="config file for Experiment"
        )
        parser.add_argument(
            "--project-path",
            default=os.getcwd(),
            help="path to store matrices and trained models",
        )
        parser.add_argument(
            "--n-db-processes",
            type=natural_number,
            default=1,
            help="number of concurrent database connections to use",
        )
        parser.add_argument(
            "--n-processes",
            type=natural_number,
            default=1,
            help="number of cores to use",
        )
        parser.add_argument(
            "--matrix-format",
            choices=self.matrix_storage_map.keys(),
            default=self.matrix_storage_default,
            help=f"The matrix storage format to use. [default: {self.matrix_storage_default}]"
        )
        parser.add_argument("--replace", dest="replace", action="store_true")
        parser.add_argument(
            "-v",
            "--validate",
            action="store_true",
            help="validate before running experiment",
        )
        parser.add_argument(
            "--no-validate",
            action="store_false",
            dest="validate",
            help="run experiment without validation",
        )
        parser.add_argument(
            "--validate-only",
            action="store_true",
            help="only validate the config file not running Experiment",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            dest="profile",
            help="Record the time spent in various functions using cProfile"
        )

        parser.add_argument(
            "--no-materialize-fromobjs",
            action="store_false",
            dest="materialize_fromobjs",
            help="do not attempt to create tables out of any feature 'from obj' subqueries."
        )

        parser.add_argument(
            "--save-predictions",
            action="store_true",
            dest="save_predictions",
            default=True,
            help="Save predictions in the database to enable more analyses after modeling [default: True]",
        )

        parser.add_argument(
            "--no-save-predictions",
            action="store_false",
            default=True,
            dest="save_predictions",
            help="Skip saving predictions to the database to save time",
        )

        parser.add_argument(
            "--features-ignore-cohort",
            action="store_true",
            default=False,
            dest="features_ignore_cohort",
            help="Will save all features independently of cohort. " +
            "This can require more disk space but allow you to reuse " +
            "features across different cohorts"
        )

        parser.add_argument(
            "--show-timechop",
            action="store_true",
            default=False,
            help="Visualize time chops (temporal cross-validation blocks')"
        )

        parser.set_defaults(validate=False, validate_only=False, materialize_fromobjs=True)

    def _load_config(self):
        config_file = Store.factory(self.args.config)
        return yaml.load(config_file.load())

    @cachedproperty
    def experiment(self):
        self.root.setup()  # Loading configuration (if exists)
        db_url = self.root.db_url
        config = self._load_config()
        db_engine = create_engine(db_url)
        common_kwargs = {
            "db_engine": db_engine,
            "project_path": self.args.project_path,
            "config": config,
            "replace": self.args.replace,
            "materialize_subquery_fromobjs": self.args.materialize_fromobjs,
            "features_ignore_cohort": self.args.features_ignore_cohort,
            "matrix_storage_class": self.matrix_storage_map[self.args.matrix_format],
            "profile": self.args.profile,
            "save_predictions": self.args.save_predictions
        }
        if self.args.n_db_processes > 1 or self.args.n_processes > 1:
            experiment = MultiCoreExperiment(
                n_db_processes=self.args.n_db_processes,
                n_processes=self.args.n_processes,
                **common_kwargs,
            )
        else:
            experiment = SingleThreadedExperiment(**common_kwargs)
        return experiment

    def __call__(self, args):
        if args.validate_only:
            self.experiment.validate()
        elif args.validate:
            self.experiment.validate()
            self.experiment.run()
        elif args.show_timechop:
            experiment_name = os.path.splitext(os.path.basename(self.args.config.name))[0]
            project_storage = self.experiment.project_storage
            timechop_store = project_storage.get_store(
                ["images"],
                f"{experiment_name}.png"
                )

            with timechop_store.open('wb') as fd:
                visualize_chops(self.experiment.chopper, save_target=fd)

        else:
            self.experiment.run()


@Triage.register
class Audition(Command):
    """Audition models from a completed experiment to pick a smaller group of promising models
    """

    def __init__(self, parser):
        parser.add_argument(
            "-c",
            "--config",
            type=argparse.FileType("r"),
            default="audition_config.yaml",
            help="config file for audition",
        )
        parser.add_argument(
            "-v",
            "--validate",
            action="store_true",
            help="validate before running audition",
        )
        parser.add_argument(
            "--no-validate",
            action="store_false",
            dest="validate",
            help="run audition without validation",
        )
        parser.add_argument(
            "--validate-only",
            action="store_true",
            help="only validate the config file not running audition",
        )
        parser.add_argument(
            "-d",
            "--directory",
            help="directory to store the result plots from audition",
        )
        parser.set_defaults(directory=None, validate=True, validate_only=False)

    @cachedproperty
    def runner(self):
        self.root.setup()  # Loading configuration (if exists)
        db_url = self.root.db_url
        dir_plot = self.args.directory
        config = yaml.load(self.args.config)
        db_engine = create_engine(db_url)
        return AuditionRunner(config, db_engine, dir_plot)

    def __call__(self, args):
        if args.validate_only:
            self.runner.validate()
        elif args.validate:
            self.runner.validate()
            self.runner.run()
        else:
            self.runner.run()


@Triage.register
class Db(Command):
    """Manage experiment database"""

    @cmdmethod
    def upgrade(self, args):
        """Upgrade triage results database"""
        upgrade_db(args.dbfile)

    @cmdmethod(
        "configversion",
        choices=REVISION_MAPPING.keys(),
        help="config version of last experiment you ran",
    )
    def stamp(self, args):
        """Instruct the triage results database to mark itself as updated to a
        known version without doing any upgrading.

        Use this if the database was created without an 'alembic_version' table.
        Uses the config version of your experiment to infer what database version is suitable.
        """
        revision = REVISION_MAPPING[args.configversion]
        print(
            f"Based on config version {args.configversion} "
            f"we think your results schema is version {revision} and are upgrading to it"
        )
        stamp_db(revision, args.dbfile)


def execute():
    main(Triage)


if __name__ == "__main__":
    main(Triage)
