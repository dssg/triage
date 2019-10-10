# Upgrading your experiment configuration to v7


This document details the steps needed to update a triage v6 configuration to
v7, mimicking the old behavior.

Experiment configuration v7 includes only one change from v6: the addition of a mandatory random_seed, that is set at the beginning of the experiment and affects all subsequent random numbers. It is expected to be an integer.

Old:
```yaml

config_version: 'v6'

# EXPERIMENT METADATA
```

New:
```yaml

config_version: 'v7'

# EXPERIMENT METADATA
# random_seed will be set in Python at the beginning of the experiment and 
# affect the generation of all model seeds
random_seed: 23895478
```
