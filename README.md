Triage
======

Risk modeling and prediction for public policy

[![image](https://travis-ci.org/dssg/triage.svg?branch=master)](https://travis-ci.org/dssg/triage)
[![image](https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/triage)
[![image](https://codeclimate.com/github/dssg/triage.png)](https://codeclimate.com/github/dssg/triage)

Predictive analytics projects require the coordination of many different
tasks, such as feature generation, classifier training, evaluation, and
list generation. These tasks are complicated in their own right, but in
addition have to be combined in different ways throughout the course of
the project.

Triage aims to provide interfaces to these different phases of a
project, such as an `Experiment`. Each phase is defined by configuration
specific to the needs of the project, and an arrangement of core data
science components that work together to produce the output of that
phase.

`Experiment` (create features and models) -> `Audition` (pick the best models) -> `Postmodeling` (dive into best models)

## Documentation Quick Links

- [Dirty Duck Tutorial](https://dssg.github.io/triage/dirtyduck/docs/)
- [Triage Documentation Site](https://dssg.github.io/triage/)

## Getting Started

### Prerequisites

To use Triage, you first need:

- Python 3.6
- A PostgreSQL database with your source data (events, geographical data, etc) loaded.
- Ample space on an available disk, (or for example in Amazon Web Services's S3), to store the needed matrices and models for your experiments
- A question you want to answer!

### Install

Triage is a Python package distributable via `setuptools`. It may be
installed directly using `easy_install` or `pip` (`pip install triage`), or named as a
dependency of another package as `triage`.

### Design an Experiment

The first thing you can do in Triage is run an experiment (the later phases rely on Experiment results to work). Triage experiments require a lot of configuration. You can see some [sample configuration with explanations](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) to see what configuration looks like. But if you're new to Triage, you will be much better off [reading the Dirty Duck tutorial](https://dssg.github.io/triage/dirtyduck/docs/) as opposed to jumping into the config file.

### Run an Experiment

Once you've defined your experiment, you can run it from the command-line or from within a Python program.

The Triage CLI defaults database connection information to a file stored in 'database.yaml' (example in [example_database.yaml](example_database.yaml)), so with this you can omit any mention of the database.

CLI:
```bash

triage experiment example/config/experiment.yaml
```

Python:
```python
from triage.experiments import SingleThreadedExperiment

experiment = SingleThreadedExperiment(
    config=experiment_config, # a dictionary
    db_engine=create_engine(...), # http://docs.sqlalchemy.org/en/latest/core/engines.html
    project_path='/path/to/directory/to/save/data'
)
experiment.run()
```

There are a plethora of options available for experiment running, affecting things like parallelization, storage, and more. These options are detailed in the [Running an Experiment](https://dssg.github.io/triage/experiments/running/) page.


## Triage Phase Details

### Experiment

> I have a bunch of data and a question I want to answer. How do I answer the question?

An experiment represents the initial research work of creating design matrices from source data, and training/testing/evaluating a model grid on those matrices. At the end of the experiment, a relational database with results metadata is populated, allowing for evaluation by the researcher. 


If you're new to Triage Experiments, check out the [Dirty Duck tutorial](https://dssg.github.io/dirtyduck). It's a guided tour through Triage functionality using a real-world problem.

If you're familiar with creating an Experiment but want to see more reference documentation and some deep dives, check out the links on the side.

### Audition

> I just trained a bunch of models. How do I pick the best ones?

Audition is a tool for picking the best trained classifiers from a predictive analytics experiment. Often, production-scale experiments will come up with thousands of trained models, and sifting through all of those results can be time-consuming even after calculating the usual basic metrics like precision and recall. Which metrics matter most? Should you prioritize the best metric value over time or treat recent data as most important? Is low metric variance important? The answers to questions like these may not be obvious up front. Audition introduces a structured, semi-automated way of filtering models based on what you consider important, with an interface that is easy to interact with from a Jupyter notebook (with plots), but is driven by configuration that can easily be scripted.

To get started with Audition, check out its [README](https://github.com/dssg/triage/tree/master/src/triage/component/audition)

### Postmodeling

> What is the distribution of my scores? What is generating a higher FPR in model x compared to model y? What is the single most important feature in my models?`

This questions, and other ones, are the kind of inquiries that the triage user may have in mind when scrolling trough the models selected by the Audition component. Choosing the right model for deployment and exploring its predictions and behavior in time is a pivotal task. postmodeling will help to answer some of this questions by exploring the outcomes of the model, and exploring "deeply" into the model behavior across time and features.

[Get started with Postmodeling](https://github.com/dssg/triage/tree/master/src/triage/component/postmodeling/contrast)


Background
----------

Triage is developed at the University of Chicago's [Center For Data
Science and Public Policy](http://dsapp.uchicago.edu). We created it in
response to commonly occuring challenges we've encountered and patterns
we've developed while working on projects for our partners.


## Development
To build this package (without installation), its dependencies may
alternatively be installed from the terminal using `pip`:

    pip install -r requirement/main.txt

### Testing

To add test (and development) dependencies, use **test.txt**:

    pip install -r requirement/test.txt [-r requirement/dev.txt]

Then, to run tests:

    pytest

### Development

To quickly bootstrap a development environment, having cloned the
repository, invoke the executable `develop` script from your system
shell:

    ./develop

A "wizard" will suggest set-up steps and optionally execute these, for
example:

    (install) begin

    (pyenv) installed

    (python-3.6.2) installed

    (virtualenv) installed

    (activation) installed

    (libs) install?
    1) yes, install {pip install -r requirement/main.txt -r requirement/test.txt -r requirement/dev.txt}
    2) no, ignore
    #? 1

### Contributing

If you'd like to contribute to Triage development, see the [CONTRIBUTING.md](CONTRIBUTING.md) document.

