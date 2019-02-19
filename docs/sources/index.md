# Triage

Risk modeling and prediction

[![Build Status](https://travis-ci.org/dssg/triage.svg?branch=master)](https://travis-ci.org/dssg/triage)
[![codecov](https://codecov.io/gh/dssg/triage/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/triage)
[![codeclimate](https://codeclimate.com/github/dssg/triage.png)](https://codeclimate.com/github/dssg/triage)


Predictive analytics projects require the coordination of many different tasks, such as feature generation, classifier training, evaluation, and list generation. These tasks are complicated in their own right, but in addition have to be combined in different ways throughout the course of the project. 

Triage aims to provide interfaces to these different phases of a project, such as an `Experiment`. Each phase is defined by configuration specific to the needs of the project, and an arrangement of core data science components that work together to produce the output of that phase.

<script>mermaid.initialize({startOnLoad:true});</script>

The phases currently implemented in Triage are:

<div class="mermaid">
graph LR
    Experiment["Experiment (create features and models)"]
    Audition["Audition (pick the best models)"]
    Postmodeling["Postmodeling (dive into best models)"]

    Experiment --> Audition
    Audition --> Postmodeling
</div>

### Experiment

>> I have a bunch of data. How do I create some models?

An experiment represents the initial research work of creating design matrices from source data, and training/testing/evaluating a model grid on those matrices. At the end of the experiment, a relational database with results metadata is populated, allowing for evaluation by the researcher.


If you're new to Triage Experiments, check out the [Dirty Duck tutorial](https://dssg.github.io/dirtyduck). It's a guided tour through Triage functionality using a real-world problem.

If you're familiar with creating an Experiment but want to see more reference documentation and some deep dives, check out the links on the side.

## Audition

>> I just trained a bunch of models. How do I pick the best ones?

Audition is a tool for picking the best trained classifiers from a predictive analytics experiment. Often, production-scale experiments will come up with thousands of trained models, and sifting through all of those results can be time-consuming even after calculating the usual basic metrics like precision and recall. Which metrics matter most? Should you prioritize the best metric value over time or treat recent data as most important? Is low metric variance important? The answers to questions like these may not be obvious up front. Audition introduces a structured, semi-automated way of filtering models based on what you consider important, with an interface that is easy to interact with from a Jupyter notebook (with plots), but is driven by configuration that can easily be scripted.

To get started with Audition, check out its [README](https://github.com/dssg/triage/tree/master/src/triage/component/audition)

## Postmodeling

>> What is the distribution of my scores? What is generating a higher FPR in model x compared to model y? What is the single most important feature in my models?`

This questions, and other ones, are the kind of inquiries that the triage user may have in mind when scrolling trough the models selected by the Audition component. Choosing the right model for deployment and exploring its predictions and behavior in time is a pivotal task. postmodeling will help to answer some of this questions by exploring the outcomes of the model, and exploring "deeply" into the model behavior across time and features.

To get started with Postmodeling, check out its [README](https://github.com/dssg/triage/tree/master/src/triage/component/postmodeling/contrast)


## Background

Triage is developed at the University of Chicago's [Center For Data Science and Public Policy](http://dsapp.uchicago.edu). We created it in response to commonly occuring challenges we've encountered and patterns we've developed while working on projects for our partners.
