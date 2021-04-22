# -*- coding: utf-8 -*-

__author__ = """Center for Data Science and Public Policy"""
__email__ = "datascifellows@gmail.com"
__version__ = '4.3.1' # do not change to double-quotes, it will screw up bumpversion

import logging
import logging.config
import yaml
import pathlib


logging_config = pathlib.Path(__file__).parent / 'config' / 'logging.yaml'

with open(logging_config, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


from .util.db import create_engine



__all__ = ('create_engine',)
