# Description of the problem

This tutorial aims to introduce the reader to
[Triage](https://github.com/dssg/triage), a machine learning modeling
tool built by the [Center for Data Science and Public
Policy](https://dsapp.uchicago.edu). We will use the well-known
[Chicago Food Inspections
dataset](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5)[^1].

We will present the two problems that `triage` was built to model[^2]:

1.  [**Resource Prioritization Systems**](inspections.md) (also known as an *inspections problem*)[^3] and
2.  [**Early Warning Systems**](eis.md)[^4].


## Resource Prioritization Systems

In an ideal world, inspectors would visit every food
facility, every day[^5] to ensure it meets safety standards. But the
real world doesn't have enough inspectors for that to happen, so the
city needs to decide how to allocate its limited inspection workforce
to find and remediate as many establishments with food hazards as
possible. Assuming the city can inspect $n$ facilities in the next
$X$ period of time, they can define the problem as:

> Which $n$ facilities will have a food violation in the following $X$ period of time?

If our inspection workforce is really limited, we should probably just
target the most serious violations. Then we'd define the problem as:

> Which $n$ facilities will have a critical or serious violation in the following $X$ period of time?

The answer to this question is a list of length $n$ with the
facilities at high risk of found a violation if they are inspected in
the following $X$ period of time.

If you want to continue to this case studie click [here](inspections.md)

## Early Warning Systems

Using the same dataset (*Chicago Food Inspections dataset*), facility
owners or managers would pose the
machine learning (ML) problem as an early warning problem. They'd like
to know whether an inspector is going to visit their facility so they
can prepare for it. They can define the problem as:

> Will my facility be inspected in the next $X$ period of time?

If you want to continue to this case studie click [here](eis.md)

!!! note "Important"
    Note that in both case studies, *resource prioritization* and
    *early warning* systems we are defining a period of time in which
    the event will potentially happen. This makes this ML problem a
    _prediction_ problem instead of a _classification_ problem.


## What do they have in common?

For either problem, $X$ could be a day, a week, month, a quarter, a
year, 56 days, or some other time period.

Without going into much detail, both problems use data where each row
describes an **event** in which an **entity** was involved, and each
event has a specific **outcome** or result.

The **entity** for both inspection prioritizations and early warnings
in this tutorial is a food *facility*, and the **event** is an
inspection. But the **outcome** differs. For inspections the outcome
is whether the *inspection failed* or *major violation was found*, while for early
warning the outcome is whether the facility was *inspected*.

## How do they differ?

Besides the obvious (e.g. the label), these ML's problem formulations have
very different internal structure:

Fot the *EIS* problem **all** of the entities of interest in a given
period of time **have** a label. The *Inspections* problem does not
have that luxury. Given all of the existing entities of interest only a
fraction are *inspected* which means that only the inspected
facilities will have a label (`True/False`) since these are the only
entities with a known outcome (e.g a major violation was discovered
during the inspection), but all of the remaining ones
will not have a label. This
will be reflected, in the *training* matrices since you only
train on the facilities that were inspected (so you will have less
rows in them). Another impact will be in the metrics. You will need to be
very careful about interpreting the metrics in an inspections
problem. Finally, when you are designing the field validation of your
model, you need to take in account **selection bias**. If not, you
will be inspecting the same facilities over and over and never inspect any facilities you have not inspected before.


## What's next?

- Learn more about [early warning systems](eis.md)
- Learn more about [resource prioritization systems](inspections.md)


[^1]: Several examples use this dataset, such as [City of Chicago Food Inspection Forecasting](https://chicago.github.io/food-inspections-evaluation/), [PyCon 2016 keynote: Built in Super Heroes](https://youtu.be/lyDLAutA88s), and [PyData 2016: Forecasting critical food violations at restaurants using open data](https://youtu.be/1dKonIT-Yak).

[^2]: It is also possible to do "visit-level prediction" type of ML problem.

[^3]: Examples include [Predictive Enforcement of Hazardous Waste Regulations](http://www.datasciencepublicpolicy.org/projects/energy-and-environment/) and [Targeting Proactive Inspections for Lead Hazards](http://www.datasciencepublicpolicy.org/projects/public-health/poison-prevention/).

[^4]: Examples include [Increasing High School Graduation Rates: Early Warnings and Predictive Systems](http://www.datasciencepublicpolicy.org/projects/education/), [Building Data-Driven Early Intervention Systems for Police Officers](http://www.datasciencepublicpolicy.org/projects/public-safety/), and [Data-Driven Justice Initiative: Identifying Frequent Users of Multiple Public Systems for More Effective Early Assistance](http://www.datasciencepublicpolicy.org/projects/criminal-justice/).

[^5]: Defined as "bakery, banquet hall, candy store, caterer, coffee shop, day care center (for ages less than 2), day care center (for ages 2 – 6), day care center (combo, for ages less than 2 and 2 – 6 combined), gas station, Golden Diner, grocery store, hospital, long term care center(nursing home), liquor store, mobile food dispenser, restaurant, paleteria, school, shelter, tavern, social club, wholesaler, or Wrigley Field Rooftop" ([source](https://data.cityofchicago.org/api/views/4ijn-s7e5/files/O9cwLJ4wvxQJ2MirxkNzAUCCMQiM31DMzRkckMsKlxc?download=true&filename=foodinspections_description.pdf)).

