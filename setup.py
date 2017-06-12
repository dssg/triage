#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('LICENSE') as license_file:
    license = license_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

with open('requirements_dev.txt') as requirements_dev_file:
    test_requirements = requirements + requirements_dev_file.readlines()

setup(
    name='results_schema',
    version='1.1.0',
    description="Store results of modeling runs",
    long_description=readme,
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/results-schema',
    packages=[
        'results_schema',
        'results_schema.alembic',
        'results_schema.alembic.versions',
    ],
    include_package_data=True,
    install_requires=requirements,
    tests_require=test_requirements,
    license=license,
    zip_safe=False,
    keywords='analytics datascience modeling modelevaluation',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
)
