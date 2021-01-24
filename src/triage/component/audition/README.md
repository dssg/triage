# Audition

Choosing the best classifier models

[![Build Status](https://travis-ci.org/dssg/audition.svg?branch=master)](https://travis-ci.org/dssg/audition)
[![codecov](https://codecov.io/gh/dssg/audition/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/audition)
[![codeclimate](https://codeclimate.com/github/dssg/audition.png)](https://codeclimate.com/github/dssg/audition)

## Overview
Audition is a tool for picking the best trained classifiers from a predictive analytics experiment. Often, production-scale experiments will come up with thousands of trained models, and sifting through all of those results can be time-consuming even after calculating the usual basic metrics like precision and recall. Which metrics matter most? Should you prioritize the best metric value over time or treat recent data as most important? Is low metric variance important? The answers to questions like these may not be obvious up front. Audition introduces a structured, semi-automated way of filtering models based on what you consider important, with an interface that is easy to interact with from a Jupyter notebook (with plots), but is driven by configuration that can easily be scripted.

**Find documentation for Audition [here](https://dssg.github.io/triage/audition/audition_intro/)**

### Use Case
We’ve just built hundreds of models over time - each model group trains on a given train set and scores the test set and calculates several metrics.

### What does it do?
**Input**:

* List of model groups built over different train periods and tested on different test periods (train end times)
	* The train end times for each model group should be the same as the list or subset of the list, otherwise those models with unmatched train end times would be pruned out in the first round.
* Model selection rules
* Metric(s) of interest
* (Optional) Method of aggregating metrics when multiple models exist for a given `model_group_id` and `train_end_time` combination (e.g., from different random seeds) -- `mean`, `best`, or `worst` (the default)

**Process**:

1. Gets rid of really bad model groups wrt the metric of interest. A model group is discarded if:
	* It’s never close to the “best” model (define close to best) or
	* If it’s metric is below a certain number (define min threshold)  at least once

We iterate over various values of these parameters to end up with a reasonable number of model groups to pass ot the next step

2. Apply existing (or new, custom) selection rules to the model groups passed from step 1. Current supported rules are “best_most_recent”, “best_average” for one or two metrics, best_recent_average, “lowest_variance”, high_avg_low_variance, “most_frequent_best_distance”
    1. For each rule
        1. For each time period
            1. It applies the rule to select a model for that time period (based on performance of that model so far)
            2. It calculates the evaluation metric for that selected model on the next time period
            3. calculates regret (how much worse is the selected model compared to the best model in the next time period)
                1. Absolute value
                2. rank/percentile [todo]

3. Now we have a regret for each rule for each time period. We now have to decide on which rule to use. Do all/most rules pick the same/similar model groups? If so, then audition should output those model groups

Output:
* Top k model groups for each rule and average regret for each rule