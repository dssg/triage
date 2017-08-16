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
    test_requirements = requirements + [
        line for line in dev_requirements_file.readlines()
        if '-r requirements' not in line
    ]


setup(
    name='matrix-architect',
    version='0.2.1',
    description="Plan, design, and build train and test matrices",
    long_description=readme,
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/architect',
    packages=[
        'architect',
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
    ],
    test_suite='tests',
    tests_require=test_requirements
)
