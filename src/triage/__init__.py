# -*- coding: utf-8 -*-

__author__ = """Center for Data Science and Public Policy"""
__email__ = "datascifellows@gmail.com"
__version__ = '5.5.5' # do not change to double-quotes, it will screw up bumpversion

from .logging import configure_logging


configure_logging()


from .util.db import create_engine



__all__ = ('create_engine',)
