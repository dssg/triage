# -*- coding: utf-8 -*-

__author__ = """Center for Data Science and Public Policy"""
__email__ = "datascifellows@gmail.com"
__version__ = '3.3.0' # do not change to double-quotes, it will screw up bumpversion


from .util.db import create_engine

__all__ = ('create_engine',)
