#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('LICENSE') as license_file:
    license = license_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

with open('requirements_dev.txt') as dev_requirements_file:
    test_requirements = [
        line for line in dev_requirements_file.readlines()
        if '-r requirements' not in line
    ]


setup(
    name='model-catwalk',
    version='0.2.1',
    description="Training, testing, and evaluating machine learning classifier models",
    long_description=readme,
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/catwalk',
    packages=[
        'catwalk',
        'catwalk.estimators',
        'catwalk.individual_importance'
    ],
    include_package_data=True,
    install_requires=requirements,
    license=license,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
