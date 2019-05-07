# Welcome!

This tutorial will show you how to use `Triage`, a data science
modeling tool developed at the [Center for Data Science and Public
Policy](http://dsapp.uchicago.edu) (DSaPP) at the University of
Chicago.

`Triage` helps build models for two [common applied
problems](https://dssg.uchicago.edu/data-science-for-social-good-conference-2017/training-workshop-data-science-for-social-good-problem-templates/):
(a) Early warning systems (**EWS** or **EIS**), (b) *resource
prioritization* (a.k.a "an inspections problem") . These problems are
difficult to model because their conceptualization and and
implementation are prone to error, given their multi-dimensional,
multi-entity, time-series structure.

**NOTE** This tutorial is in sync with the latest version of
`triage`. At this moment
[v4.0.0](https://github.com/dssg/triage/releases/tag/v4.0.0).


# Before you start


## What you need for this tutorial

Install [Docker CE](http://www.docker.com) and [Docker
Compose](https://docs.docker.com/compose/). That's it! Follow the
links for the installation instructions.

Note that if you are using `GNU/Linux` you should add your user to the
`docker` group following the instructions at this
[link](https://docs.docker.com/install/linux/linux-postinstall/).

At the moment only operative systems with \*nix-type command lines are
supported, such as `GNU/Linux` and `MacOS`. Recent versions of
`Windows` may also work.


## How to use this tutorial

First, clone this repository on your laptop

    git clone https://github.com/dssg/triage

Second, in the `triage/docs/sources/dirtyduck/` directory run

    ./tutorial.sh up

This will take several minutes the first time you do it.


## How you can help to improve this tutorial

If you want to contribute, please follow the suggestions in the triage’s (github repository)[https://github.com/dssg/triage/tree/master/docs/sources/dirtyduck].
