# coding: utf-8

from setuptools import setup

setup(
    name='triage_experiment',
    version='0.1',
    py_modules=['triage_experiment'],
    entry_points='''
        [console_scripts]
        triage_experiment=triage_experiment:triage
    ''',
)
