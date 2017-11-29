CONFIG_VERSION = 'v3'

from .base import ExperimentBase
from .multicore import MultiCoreExperiment
from .singlethreaded import SingleThreadedExperiment

__all__ = (
    'ExperimentBase',
    'MultiCoreExperiment',
    'SingleThreadedExperiment',
)
