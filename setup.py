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
    name='triage',
    version='1.1.1',
    description="Risk modeling and prediction",
    long_description=readme,
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/triage',
    packages=[
        'triage',
        'triage.experiments',
    ],
    package_dir={'triage':
                 'triage'},
    include_package_data=True,
    install_requires=requirements,
    license=license,
    zip_safe=False,
    keywords='triage',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
