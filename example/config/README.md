### Triage Example Config Files

This folder contains examples of the config files that control Triage. These config files exist to demonstrate the format and syntax of Triage's config files, and provide templates for implementing new projects in triage.

#### audition.yaml

An example of the config file that controls Audition, the Triage model selection module. Find additional documentation for the Audition config file [here](https://dssg.github.io/triage/dirtyduck/audition/audition-config/).

#### database.yaml

Triage requires a database connection for source data and [model governance](https://dssg.github.io/triage/dirtyduck/ml_governance/). Use a file of this format to specify your connection.

#### dirty-duckling.yaml

A Triage experiment config file used in [Dirty Duckling](https://dssg.github.io/triage/dirtyduck/), the Triage tutorial.

#### experiment.yaml

An example of an experiment config file. Experiment configs control behavior of the Triage experiment pipeline, which handles feature and label generation, model training, and model evaluation. Find more documentation for the Triage experiment config file [here](https://dssg.github.io/triage/experiments/experiment-config/).

#### postmodeling_config.yaml & postmodeling_crosstabs.yaml

Controls the Triage Postmodeling module. Postmodeling is currently under development. It provides a set of tools for evaluating and investigating trained models. More documentation is available [here](https://dssg.github.io/triage/postmodeling/postmodeling-config).