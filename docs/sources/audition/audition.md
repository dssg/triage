# Model selection

How to pick the best one and use it for making predictions with *new* data? What do you mean by “the best”? This is not as easy as it sounds, due to several factors:

-   You can try to pick the best using a metric specified in the config file (`precision@` and `recall@`), but at what point of time? Maybe different model groups are best at different prediction times.
-   You can just use the one that performs best on the last test set.
-   You can value a model group that provides consistent results over time. It might not be the best on any test set, but you can feel more confident that it will continue to perform similarly.
-   If there are several model groups that perform similarly and their lists are more or less similar, maybe it doesn't really matter which you pick.

The answers to questions like these may not be obvious up front. Let’s discuss an imaginary example, that will help to clarify this<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>

![img](images/sanjose-2.png "A simplified example of our model evaluation process: three different models are trained using information prior to 2014, 2015, and 2016 and evaluated on what actually happened in those years. Looking at how each model performs over time allows us to balance stability and performance. (From Data-Driven Inspections for Safer Housing in San Jose, California)")

The graphic above provides an imaginary example with three three different *models groups* on how they performed against the actual results of inspections in 2014, 2015, and 2016. In the $x$-axis, the date of the predictions, inn the $y$-axes the metric of interest (e.g. *precision*, *recall*, etc). Recall that the *models groups* differ in a number of ways, for instance: including or excluding different types of *features*, employing different algorithms or hyperparameters, or focusing on more or less recent information, this comes from *timechop configuration*, (e.g *how much past data* do you want to include).

We start by building a set of models that seek to predict what will happen in 2014, using only information that would have been available to inspectors before January 1, 2014. We then evaluate the models based on what actually happened in 2014. Repeating this process for 2015 and 2016 gives us an idea of how well — and how dependably — a given model is able to predict the future.

Now, for the selection you could have a reasoning process as follows:

-   You can probably eliminate the yellow triangle model right off the bat.
-   If we only looked at 2016, we’d choose the light blue squares model, but although it does well in 2016, it performed the worst in 2015, so we don’t know if we can trust its performance – what if it dips back down in 2017? Then again, what if 2015 was just some sort of anomaly? We don’t know the future (which is why we need analysis like this), but we want to give ourselves the best advantage we can.
-   To balance consistency and performance, we choose a model that reliably performs well (blue circles), even if it’s not always the best.

In this particular/imaginary example , the “selection process” was kind of easy. Of course,in real life we were choosing between more than three models; we just built and evaluated more than **46** *models groups*!

Remember, this is only one way of choose! You could have (or better, the organization could have) a different opinion about what consists a *best* model.

Triage provides this functionality in `audition` and in `postmodel`. At the moment of this writing, these two modules require more interaction (i.e. they aren't integrated with the *configuration file*).

Audition is a tool for picking the best trained classifiers from a predictive analytics experiment. Audition introduces a structured, semi-automated way of filtering models based on what you consider important.

`Audition` formalizes this idea through *selection rules* that take in the data up to a given point in time, apply some rule to choose a model group, and then evaluate the performance (**regret**) of the chosen model group in the subsequent time window.

`Audition` predefines 7 rules:

1.  `best_current_value` :: Pick the model group with the best current metric Value.
2.  `best_average_value` :: Pick the model with the highest average metric value so far.
3.  `lowest_metric_variance` :: Pick the model with the lowest metric variance so far.
4.  `most_frequent_best_dist` :: Pick the model that is most frequently within `dist_from_best_case` from the best-performing model group across test sets so far.
5.  `best_average_two_metrics` :: Pick the model with the highest average combined value to date of two metrics weighted together using `metric1_weight`.
6.  `best_avg_var_penalized` :: Pick the model with the highest average metric value so far, penalized for relative variance as: \[ =avg_value - (stdev_penalty) * (stdev - min_stdev)= \] where `min_stdev` is the minimum standard deviation of the metric across all model groups
7.  `best_avg_recency_weight` :: Pick the model with the highest average metric value so far, placing less weight in older results. You need to specify two parameters: the shape of how the weight affects points (`decay_type`, linear or exponential) and the relative weight of the most recent point (`curr_weight`).

> Before move on, remember the two main *caveats* for the value of the metric in this kind of ML problems:
>
> -   Could be many entities with the same predicted risk score (*ties*)
> -   Could be a lot of entities without a label (Weren't inspected, so we don’t know)

We included a [simple configuration file](file:///home/nanounanue/projects/dsapp/dirtyduck/triage/inspection_audition_config.yaml) with some rules:

```yaml
# CHOOSE MODEL GROUPS
model_groups:
    query: |
        select distinct(model_group_id)
        from model_metadata.model_groups
        where model_config ->> 'experiment_type' ~ 'inspection'
# CHOOSE TIMESTAMPS/TRAIN END TIMES
time_stamps:
    query: |
        select distinct train_end_time
        from model_metadata.models
        where model_group_id in ({})
        and extract(day from train_end_time) in (1)
        and train_end_time >= '2015-01-01'
# FILTER
filter:
    metric: 'precision@' # metric of interest
    parameter: '10_pct' # parameter of interest
    max_from_best: 1.0 # The maximum value that the given metric can be worse than the best model for a given train end time.
    threshold_value: 0.0 # The worst absolute value that the given metric should be.
    distance_table: 'inspections_distance_table' # name of the distance table
    models_table: 'models' # name of the models table

# RULES
rules:
    -
        shared_parameters:
            -
                metric: 'precision@'
                parameter: '10_pct'

        selection_rules:
            -
                name: 'best_current_value' # Pick the model group with the best current metric value
                n: 3
            -
                name: 'best_average_value' # Pick the model with the highest average metric value
                n: 3
            -
                name: 'lowest_metric_variance' # Pick the model with the lowest metric variance
                n: 3
            -
                name: 'most_frequent_best_dist' # Pick the model that is most frequently within `dist_from_best_case`
                dist_from_best_case: [0.05]
                n: 3

```

`Audition` will have each rule give you the best \(n\) *model groups* based on the metric and parameter following that rule for the most recent time period (in all the rules shown \(n\) = 3).

We can run the simulation of the rules against the experiment as:

```sh
# Run this in bastion…
triage --tb audition -c inspection_audition_config.yaml --directory audition/inspections
```

`Audition` will create several plots that will help you to sort out which is the *best* model group to use (like in a production setting or just to generate your predictions list).

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> The discussion was ~~taken~~ stolen from [Data-Driven Inspections for Safer Housing in San Jose, California](https://dssg.uchicago.edu/2017/07/14/data-driven-inspections-for-safer-housing-in-san-jose-california/) (Kit Rodolfa, Jane Zanzig 2016) Great read by the way!
