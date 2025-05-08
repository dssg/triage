This is a guide to `Triage`, a machine learning / data science tool initially developed at the [Center for Data Science and Public
Policy](http://datasciencepublicpolicy.org) (DSaPP) at the University of
Chicago and now being maintained at Carnegie Mellon University.


`Triage` helps build ML systems for two [common
problems](https://dssgfellowship.org/data-science-for-social-good-conference-2017/training-workshop-data-science-for-social-good-problem-templates/):
(a) Early warning systems (**EWS** or **EIS**), (b) *resource
prioritization* (a.k.a "an inspections problem"). These problems require careful thought and design and their formulation and
implementation are often done incorrectly.

!!! info Triage version
    This tutorial may not be compatible  with the latest version of `triage` and was written for [v4.2.0](https://github.com/dssg/triage/releases/tag/v4.2.0). *We recommend starting with the [tutorial hosted at colab](https://colab.research.google.com/github/dssg/triage/blob/master/example/colab/colab_triage.ipynb)*

!!! info "How you can help to improve this tutorial"

    If you want to contribute, please follow the suggestions in the
    triage’s [github repository](https://github.com/dssg/triage/tree/master/docs/sources/dirtyduck).


## Why Dirty Duck??

There is a famous (and delicious) peking duck restaurant in Chicago called Sun Wah. We love that place, and as every restaurant in Chicago area, it gets
inspected, so the naming is an *homage* to them.


## Who is this tutorial for?

We created this tutorial with two roles in mind:

- Data scientists/ML practitioners who want to focus
on the problem they are tackling, and not on the nitty-gritty details about
how to configure and setup a Machine learning pipeline, model
governance, reproducibility, model selection, etc.

- analytical policy team without too deep of a technical/engineering background who want to
  learn how to formulate their policy problems as  Machine Learning
  problems.


## How to use this tutorial

First, clone this repository on your machine

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

At the moment only operating systems with \*nix-type command lines are
supported, such as `GNU/Linux` and `MacOS`. Recent versions of
`Windows` may also work.
