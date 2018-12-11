# Temporal Validation Deep Dive

Triage implements a kind of validation we call _temporal validation_ or
_temporal cross-validation_. In standard forms of cross validation, the
entities being modeled, their labels, and their features are treated as stable,
time-independent components of the modeling problem. For example, given a data
set of images to classify as containing a human face or not, the set of images,
whether or not they contain faces, and what the images are like are all
treated as unchanging over time. A photo will not suddenly have a face that it
didn't have yesterday -- or if it did, perhaps the photo's file was replaced
with a different photo or some other data or processing instability has
occurred. To model these kinds of problems, people frequently use one of many
cross-validation strategies, such a _k_-fold cross-validation or
leave-one-out cross-validation that train models on a subset of rows and test
them on a different subset.

Temporal validation, in contrast, is concerned with entities (properties,
businesses, people, etc.) whose features and label states change and who enter
and leave the system being modeled. This is particularly relevant in public
policy intervention contexts. For example, if a government agency is going to
inspect 200 housing properties this month, it wants to know which properties
are most likely to have serious violations *this month* (the positive label) so
that those violations can be found and addressed. If a property won't have a
violation until six months from now, it may be a low priority compared to a
property that is a danger to residents right now. In this way, properties move
between label states, having a positive label this month but having a negative
label next month after problems are fixed. Likewise, properties themselves move
in and out of the population being modeled (called a "cohort" in triage
parlance). Properties that were torn down last month shouldn't be inspected
this month, nor should properties that won't be built until next year. Even the
features of properties change over time. A a simple example, a building's age
changes every year. But even their other features have a temporal component.
Their recent history, for example, (e.g., their number of violations in the
three months) is frequently more telling than their distant history (e.g.,
their total number of violations ever), but which violations count as _recent_
changes every month!

Triage is built around the idea that solving these kinds of problems requires
keeping careful track of when predictions are made and models are built. It
will build out historical similations of model training/testing/validation
cycles using your data, enforcing that information from the "the future" (i.e.,
after a prediction is made) is never fed into the past (i.e., your features),
keeping models honest about what they know and when they know it. The rest of 
this guide is concerned with the intimate details of temporal validation and
its implementation and configuration in triage. Throughout, we will build up a
triage temporal configuration piece by piece.

## Finding Time Splits
Temporal validation involves two kinds of time splits: feature-label splits
and train-test splits.

### Feature-Label Splits (_as-of dates_)
Like regular cross-validation strategies, temporal
validation splits training and testing data on rows, training on one set of
rows and testing on another. However, under temporal validation, the rows are
indexed by the entity _and_ a timestamp. That timestamp, which triage calls an
_as-of date_ or _as-of time_ defines the split between the features and the
labels. The features aggregate everything we know about the entity up to that
point (in other words, everything we know _as of_ that point), and the label
designates whether the entity will belong to a class over some time period
after that date (for example, whether someone will develop a disease in
the next year, whether an inspector will find a violation at a specific
restaurant in the next month, or whether a person will have a criminal case
filed against them in the next year). This way, a model trained on such data
uses features from entities' pasts to predict what will happen to them over
some defined future time. That prediction is made on the as-of date. Finding
which _as-of dates_ to use (i.e., when to make predictions) is the primary
function of triage's temporal config and of its timechop component.

### Train-Test Splits (split dates)
As mentioned, temporal validation splits training and testing data on time. The
goal is to similate the process of training, selecting, and validating models
as if the modeling program had existed in the past. The way you might build
and implement practice is like this: You would build a model today
with outcomes you already know (for example, what happened to entities last
year) and then use it to make predictions about
the (as yet unknown) future, capturing outcomes as they occur in order to check
the model's predictions. That instant, today, between the outcomes you already
knew and trained the model on and the outcomes you haven't collected yet is 
your train-test split under temporal validation. Using historical records,
triage recreates this
process over and over again, starting trainin models in the past, then testing
them on the models' "future", repeating this process over and over and moving
forward in time for every repetition. 

### Example Narrative
To give an example of this in action, let's say you were implementing a
program to identify people as they leave prison who are likely to return to
jail in the next year so that they could be given supportive services that may
prevent re-incarceration. Every month (or every week, or every day), you have
some idea of who will be leaving prison that month. On average, about 100
inmates leave per month, and you have resources to provide interventions to
30 of them, so every month you want to know which 30 of the inmates leaving
that month are most likely to be re-arrested in the next year so you can give
your resources where they are most needed. 

So you give triage all of your historical prison release data (needed to
determine who to include in each model) and the historical arrest data (needed
to determine who has a positive or negative label) and any other data you want
to use for features (as long as it has timestamps), and you tell triage to
simulate your modeling problem from arrest data ranging from January 1, 2017
through January 1, 2019. 

Triage will start by training a model
on January 1, 2016 with features known up to that point and labels from
January 1, 2016 through December 31, 2016. This mimics your policy problem: you
want to know on January 1 (your _as-of date_) who will be arrested before the
next January 1. Triage will then use that model to make
predictions on a test set with labels from January 1, 2017 through December 31,
2017. If it uses any earlier data in the test set, you will have one form of
what we call "temporal leakage," where outcomes included in the test set (say,
arrests in December of 2016) were also used in the training set. This is a
mistake that can lead to overfitting of the model because it already "knows"
some of the outcomes it will be tested on. If you (improperly) implement
temporal validation this way, you may end up selecting a model that learns to
identify cases in the train-test overlap but will not perform well when you
implement it on truly unknown outcomes. To avoid this, *triage enforces
train-test splits where the labels in the two data sets do not overlap in
time.*

### Starting Your Temporal Config
At this point, we've already mentioned some of the concepts that go into a
triage temporal config. The amount of time aggregated into your label (e.g.,
whether it covers one day, one month, or one year) is called your
`label_timespan` by triage, and since triage allows you to configure it
separately for training and testing matrices, it is covered by two keys in
your temporal config. In the recidivism example we just went over, we would set
both to one year:

```
temporal_config:
    training_label_timespans: ['1year']
    test_label_timespans: ['1year']
```

For all temporal configuration options, triage supports any valid PostgreSQL
interval string (be warned that `1m` will be interpreted as `1 minute` by
PostgreSQL, so use it with caution; we recommend writing out the unit to avoid
confusion). Triage also lets you enter lists for most of the options in the
temporal config, which is why these options are in square brackets. (If you
leave the brackets off, triage will convert them to lists for you.) If you do
give more than one value, triage will run the crossproduct (all combinations)
of all temporal configuration parameters as part of your experiment.

We've also discussed when to start and stop aggregating data into labels. This
is controlled by the label start and end times. All data included in any label
will be from *on or after* the label start time and *before* the label end
time. For the recidivism project, we would use labels from January 1, 2016
through January 1, 2019:

```
temporal_config:
    label_start_time: '2016-01-01'
    label_stop_time: '2019-01-01'
    training_label_timespans: ['1year']
    test_label_timespans: ['1year']
```

Dates should be strings in the `YYYY-MM-DD` format.

There are analogous keys for feature start and stop times. Triage will not use
any features from before the `feature_start_time` in any feature aggregation.
This is use if, for example, you have operational data going back to the 1970s
but think it is only relevant, complete, or useful in the last 10 years. Triage
will also use the feature start and stop times in finding _as-of dates_. It
won't include any _as-of dates_ that are too early to it to generate features,
and it won't include any _as-of dates_ that are after features end, either (on
the assumption that you will not want to make predictions unless you actually
have recent data). For the recidivism example, let's assume we have (and want
to use) feature data from January 1, 2010 through January 1, 2019:

```
temporal_config:
    feature_start_time: '2016-01-01'
    feature_stop_time: '2019-01-01'
    label_start_time: '2016-01-01'
    label_stop_time: '2019-01-01'
    training_label_timespans: ['1year']
    test_label_timespans: ['1year']
```

Now, we just have to decide how often learners get retrained. As of December
2018, triage will only make predictions on the test matrix immediately
following the labels in the matrix the model was train on during a regular
experiment. In other words, for our recidicism problem, if triage trains a
model on labels covering arrests from January 1, 2016 through December 31,
2016, it will only test the model on labels covering arrests from January 1,
2017 through December 31, 2017, which
would give us our prioritized list of inmates being released in January 2017.
If we want a set of predictions for the following month to be made
automatically by our experiment, we will need to train a new model a month
later, on labels covering arrests from February 1, 2016 through January 31,
2017, with a new test set using labels covering arrests from February 1, 2017
through January 31, 2018. (We'd love to include the ability to use a model
on multiple test sets by default -- perhaps with a `prediction_frequency` 
temporal config key, and if you feel like working on this issue,
take a look at our contributing guide and get started!) To tell triage how
often to retrain the learners, use the `model_update_frequency` key in your
temporal config. For the recidivism problem, we will use one month:

```
temporal_config:
    feature_start_time: '2016-01-01'
    feature_end_time: '2019-01-01'
    label_start_time: '2016-01-01'
    label_end_time: '2019-01-01'
    model_update_frequency: '1month'
    training_label_timespans: ['1year']
    test_label_timespans: ['1year']
```

## How Many As-Of Dates Per Matrix?

We're well on our way to completing our temporal concfig at this point, but we
have just a couple more parameters to add to round it out. So far, we've only
discussed having one _as-of date_ per training and test matrix. But this is not
always what we want. We'll discuss two cases where we may want to include
additional rows with different dates and how to change the number of dates in
each matrix in triage.





A temporal validation deep dive is currently available in the Dirty Duck tutorial. [Dirty Duck - Temporal Cross-validation](https://dssg.github.io/dirtyduck/#sec-4-2-2-1)

You can produce the time graphs detailed in the Dirty Duck deep dive using the Triage CLI or through calling Python code directly. The graphs use matplotlib, so you'll need a matplotlib backend to use. Refer to the [matplotlib docs](https://matplotlib.org/faq/usage_faq.html) for more details.

## Python Code

Plotting is supported through the `visualize_chops` function, which takes a fully configured Timechop object. You may store the configuration for this object in a YAML file if you wish and load from a file, but in this example we directly set the parameters as arguments to the Timechop object. This would enable faster iteration of time config in a notebook setting.

```
from triage.component.timechop.plotting import visualize_chops
from triage.component.timechop import Timechop

chopper = Timechop(
    feature_start_time='2010-01-01'
    feature_end_time='2015-01-01'   # latest date included in features
    label_start_time='2012-01-01' # earliest date for which labels are avialable
    label_end_time='2015-01-01' # day AFTER last label date (all dates in any model are < this date)
    model_update_frequency='6month' # how frequently to retrain models
    training_as_of_date_frequencies='1day' # time between as of dates for same entity in train matrix
    test_as_of_date_frequencies='3month' # time between as of dates for same entity in test matrix
    max_training_histories=['6month', '3month'] # length of time included in a train matrix
    test_durations=['0day', '1month', '2month'] # length of time included in a test matrix (0 days will give a single prediction immediately after training end)
    training_label_timespans=['1month'] # time period across which outcomes are labeled in train matrices
    test_label_timespans=['7day'] # time period across which outcomes are labeled in test matrices
)

visualize_chops(chopper)
```

## Triage CLI

The Triage CLI exposes the `showtimechops` command which just takes a YAML file as input. This YAML file is expected to have a `temporal_config` section with Timechop parameters. You can use a full experiment config, or just create a YAML file with only temporal config parameters; the temporal config just has to be present. Here, we use the [example_experiment_config.yaml](https://github.com/dssg/triage/blob/master/example/config/experiment.yaml) from the Triage repository root as an example.

`triage showtimechops example_experiment_config.yaml`

## Result

Using either method, you should see output similar to this:

![time chop visualization](timechops.png)
