This is a guide to `Triage`, a data science workflow tool initially developed at the [Center for Data Science and Public
Policy](http://dsapp.uchicago.edu) (DSaPP) at the University of
Chicago and now being maintained at Carnegie Mellon University.

`Triage` helps build models for two [common applied
problems](https://dssg.uchicago.edu/data-science-for-social-good-conference-2017/training-workshop-data-science-for-social-good-problem-templates/):
(a) Early warning systems (**EWS** or **EIS**), (b) *resource
prioritization* (a.k.a "an inspections problem") . These problems are
difficult to model because their conceptualization and and
implementation are prone to error, given their multi-dimensional,
multi-entity, time-series structure.

!!! info Triage version
    This tutorial is in sync with the latest version of `triage`. At this moment [v4.2.0](https://github.com/dssg/triage/releases/tag/v4.2.0).

!!! info "How you can help to improve this tutorial"

    If you want to contribute, please follow the suggestions in the
    triage’s [github repository](https://github.com/dssg/triage/tree/master/docs/sources/dirtyduck).


## What's in a name?

There is a famous (and delicious) peking duck restaurant in Chicago,
we love that place, and as every restaurant in Chicago area, it gets
inspected, so the naming is an *homage* to them.


## Who is this tutorial for?

We created this tutorial with two roles in mind:

- A data scientist/ML practitioner who wants to focus
in the problem at his/her hands, not in the nitty-gritty detail about
how to configure and setup a Machine learning pipeline, Model
governance, Model selection, etc.

- A policy maker with a little of technical background that wants to
  learn how to pose his/her policy problem as a Machine Learning
  problem.


## How to use this tutorial

First, clone this repository on your laptop

    git clone https://github.com/dssg/triage

Second, in the cloned repository's top-level directory run

    ./tutorial.sh up

This will take several minutes the first time you do it.

After this, you may decide [to do the quickstart tutorial](dirty_duckling.md).


## Before you start

### What you need for this tutorial

Install [Docker CE](http://www.docker.com) and [Docker
Compose](https://docs.docker.com/compose/). That's it! Follow the
links for the installation instructions.

Note that if you are using `GNU/Linux` you should add your user to the
`docker` group following the instructions at this
[link](https://docs.docker.com/install/linux/linux-postinstall/).

At the moment only operative systems with \*nix-type command lines are
supported, such as `GNU/Linux` and `MacOS`. Recent versions of
`Windows` may also work.
