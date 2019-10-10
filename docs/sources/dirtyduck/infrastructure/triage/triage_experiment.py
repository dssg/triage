# coding: utf-8

import os
import yaml

import datetime

import click

from triage.experiments import SingleThreadedExperiment
from triage.component.catwalk.utils import filename_friendly_hash
from triage import create_engine

from utils import show_timechop, show_features_queries, show_model, audit_experiment

import logging

logging_level = logging.WARNING

logging.basicConfig(
    format="%(name)-30s  %(asctime)s %(levelname)10s %(process)6d  %(filename)-24s  %(lineno)4d: %(message)s",
    datefmt = "%d/%m/%Y %I:%M:%S %p",
    level=logging_level,
    handlers=[logging.StreamHandler()]
)

@click.group()
@click.option('--config_file', type=click.Path(),
              help="""Triage's experiment congiguration file name
                      NOTE: It's assumed that the file is located inside
                      triage/experiments)""",
              required=True)
@click.option('--triage_db', envvar='TRIAGE_DB_URL', type=click.STRING,
                help="""DB URL, in the form of 'postgresql://user:password@host_db:host_port/db',
                        by default it gets this from the environment (TRIAGE_DB_URL)""",
              required=True)
@click.option('--replace/--no-replace',
              help="Triage will (or won't) replace all the matrices and models",
              default=True)  ## Default True so it matches the default behaviour of Triage
@click.option('--debug', is_flag=True,
              help="Activate to get a lot of information in your screen")
@click.pass_context
def triage(ctx, config_file, triage_db, replace, debug):

    config_file = os.path.join(os.sep, "triage", "experiments", config_file)

    click.echo(f"Using the config file {config_file}")

    with open(config_file) as f:
        experiments = yaml.load(f)

    click.echo(f"The output (matrices and models) of this experiment will be stored in triage/output")
    click.echo(f"Using data stored in {triage_db}")
    click.echo(f"The experiment will utilize any preexisting matrix or model: {not replace}")
    click.echo(f"Creating experiment object")

    experiment = SingleThreadedExperiment(
        config=experiments,
        db_engine=create_engine(triage_db),
        project_path='/triage/output',
        cleanup=True,
        replace=replace
    )

    ctx.obj = experiment

    if debug:
        logging.basicConfig(level=logging.DEBUG)
        click.echo("Debug enabled (Expect A LOT of output at the screen!!!)")

    click.echo("Experiment loaded")

@triage.command()
@click.pass_obj
def validate(experiment):
    click.echo("Validating experiment's configuration")
    experiment.validate()

    click.echo("""
           The experiment configuration doesn't contain any obvious errors.
           Any error that occurs from now on, possibly will be related to hit the maximum
           number of columns allowed or collision in
           the column names, both due to PostgreSQL limitations.
    """)

    click.echo("The experiment looks in good shape. May the force be with you")

@triage.command()
@click.pass_obj
def run(experiment):
    start_time = datetime.datetime.now()

    click.echo("Executing experiment")
    experiment.run()
    click.echo("Done")

    end_time = datetime.datetime.now()
    click.echo(f"Experiment completed in {end_time - start_time} seconds")

@triage.command()
@click.pass_obj
def show_feature_generators(experiment):
    pass

@triage.command()
@click.pass_obj
def show_temporal_blocks(experiment):
    click.echo("Generating temporal blocks image")
    chopper = experiment.chopper
    file_name = f"{experiment.config['model_comment'].replace(' ', '_')}.svg"
    image_path=show_timechop(chopper, file_name=file_name)
    click.echo("Image stored in:")
    click.echo(image_path)
    return image_path

@triage.command()
@click.pass_obj
@click.option('--model',
              help="Model to plot",
              required=True)
def show_model_plot(experiment, model):
    click.echo("Generating model image")
    image_path = show_model(model)
    click.echo("Image stored in: ")
    click.echo(image_path)

    return image_path


@triage.command()
@click.pass_obj
@click.option('--metric',
              help="Model to plot",
              required=True)
@click.option('--rules',
              help="Path to selection rules",
              required=True)
def audit_models(experiment, metric, rules):
    click.echo("Auditing experiment")
    experiment_hash = filename_friendly_hash(experiment.config)

    with open(f"/triage/selection_rules/{rules}") as f:
        rules = yaml.load(f)

    metric, k = metric.split('@')

    audit_experiment(experiment_hash, f"{metric}@", k, rules)
