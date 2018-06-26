#!/usr/bin/env python
import yaml
from argcmdr import RootCommand, Command, main
from triage.experiments import CONFIG_VERSION
from triage.component.audition import Audition as AuditionRunner
from triage import create_engine

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


@Triage.register
class Audition(Command):
    """ Audition command to run or validate the config file
    """
    class Run(Command):
        """Run Audition
        """
        def __init__(self, parser):
            parser.add_argument(
                '-c', '--config',
                default='audition_config.yaml',
                help="config file for audition",
            )

            parser.add_argument(
                '-d', '--directory',
                default='some path',
                help="directory to store the result plots from audition",
            )

            self.db_engine = create_engine()

        def __call__(self, args):
            """A function/script to run the whole thing in Audition
            """
            dir_plot = args.directory
            config_fname = args.config
            try:
                with open(config_fname) as fd:
                    config = yaml.load(fd)
            except Exception as err:
                raise err

            AuditionRunner(config, dir_plot).run()

    class Validate(Command):
        """Validate the config file for audition
        """
        def __init__(self, parser):
            parser.add_argument(
                '-c', '--config',
                default='audition_config.yaml',
                help="config file for audition",
            )

        def __call__(self, args):
            """A function/script to run the validate the config file
            """
            config_fname = args.config
            try:
                with open(config_fname) as fd:
                    config = yaml.load(fd)
                    self.config = config
            except Exception as err:
                raise err

            AuditionRunner(config_fname).validate()


def execute():
    main(Triage)


if __name__ == '__main__':
    main(Triage)
