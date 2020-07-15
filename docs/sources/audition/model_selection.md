## Introduction

Model selection is the process of evaluating model groups trained on historical data, and selecting one to make predictions on future data. Audition provides a formal, repeatable model selection workflow, with flexibility allowing you to prioritize the parameters important to your project.

Often, a production-scale model training process will generate thousands of trained models. Sifting through all of those results can be time-consuming even after calculating the usual basic metrics like precision and recall. 

Which metrics matter most? Should you prioritize the best metric value over time or treat recent data as most important? Is low metric variance important? The answers to questions like these may not be obvious up front.

The best solution to this problem is to incorporate model selection into one's modeling pipeline, allowing the automated evaluation and selection of models for future prediction, based on their historical performance.

This article introduces the model selection process at a conceptual level, motivating the design of Audition.

!!! info "Unfinished Article"

    This article is missing some content. It could would benefit from:

    - Discussion on regret & selection rule evaluation
    - Links to postmodeling & bias

## An example model selection problem

We can examine the problem of model selection with a basic example.

![img](images/sanjose-2.png "A simplified example of our model evaluation process: three different models are trained using information prior to 2014, 2015, and 2016 and evaluated on what actually happened in those years. Looking at how each model performs over time allows us to balance stability and performance. (From Data-Driven Inspections for Safer Housing in San Jose, California)")

This plot shows the performance of three model groups on three train/test sets. The $y$-axis represents each model group's performance, on a metric like *precision* or *recall*, during each period specified by the $x$-axis.

Now that we've trained our model groups on historical data, we need to select a model group that we think will perform well on the next set of data (to be generated in 2017).

-   You can probably eliminate the yellow triangle model right off the bat.
-   If we only looked at 2016, we’d choose the light blue squares model, but although it does well in 2016, it performed the worst in 2015, so we don’t know if we can trust its performance – what if it dips back down in 2017? Then again, what if 2015 was just some sort of anomaly? We don’t know the future (which is why we need analysis like this), but we want to give ourselves the best advantage we can.
-   To balance consistency and performance, we choose a model that reliably performs well (blue circles), even if it’s not always the best.

In this imaginary example, the “selection process” was easy. Model 1 was the clear best option. Of course, in real life we are likely to have to choose between more than three models groups - Triage makes it easy to train grids of dozens or hundreds of models.

### Selection Rules
The goal of audition is to narrow a very large number of model groups to a small number of best candidates, ideally making use of the full time series of information. There are several ways one could consider doing so, using over-time averages of the metrics of interest, weighted averages to balance between metrics, the distance from best metrics, and balancing metric average values and stability.

Audition formalizes this idea through by introducing the concept of "selection rules". A selection rule is a function that:
- Takes data about the performance of a set of model groups, up to some point in time
- Ranks those models based on some criteria
- Returns `n` highest-ranked models

An ideal selection rule will always return the model group that performs best in the subsequent time period. Thus, a selection rule is evaluated by its `regret`: the difference in performance between its chosen model and the best-performing model in some time period.

You can use the [`Auditioner`](../api/audition/auditioner/#Auditioner) class to register, evaluate, and update selection rules. Audition will run simulations of different model group selection rules allowing you to assess which rule(s) is the best for your needs. 