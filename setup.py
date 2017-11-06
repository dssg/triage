#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('LICENSE') as license_file:
    license = license_file.read()


def parse_requirements(fname):
    """
    Parses requirments.txt file into a list
    Parameters
    ----------
    fname: str
        name of the file with dependencies
        (e.g., requirements.txt)
    Returns
    -------
    dependencies: ls[str]
       ls of dependencies
    """
    with open(fname, 'r') as infile:
        dependencies = (
            infile.read().splitlines()
        )

    return dependencies


setup(
    name='metta-data',
    version='1.0.0',
    description="Store/Read train and test matrices",
    long_description=readme + '\n\n',
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/metta-data',
    packages=[
        'metta',
    ],
    package_dir={'metta':
                 'metta'},
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    license=license,
    zip_safe=False,
    keywords='metta',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=parse_requirements('requirements.txt')
)
