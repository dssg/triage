# Description of the problem

This tutorial aims to introduce the reader to [triage](https://github.com/dssg/triage), a machine learning modeling tool built by the [Center for Data Science and Public Policy](https://dsapp.uchicago.edu). We will use the well-known [Chicago Food Inspections dataset](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5).<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>

We will present the two problems that `triage` was built to model<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>:

1.  **Resource prioritization** (internally known as the *inspections problem*)<sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup> and
2.  **Early warning**.<sup><a id="fnr.4" class="footref" href="#fn.4">4</a></sup>


## Inspection Prioritization

In an ideal world, inspectors would frequently visit every food facility, every day<sup><a id="fnr.5" class="footref" href="#fn.5">5</a></sup> to ensure it meets safety standards. But the real world doesn't have enough inspectors for that to happen, so the city needs to decide how to allocate its limited inspection workforce to find and remediate as many establishments with food hazards as possible. Assuming the city can inspect \(n\) facilities in the next \(X\) period of time, they can define the problem like this:

> Which \(n\) facilities will have a food violation in the following \(X\) period of time?

If our inspection workforce is really limited, we should probably just target the most serious violations. Then we'd define the problem like this:

> Which \(n\) facilities will have a critical or serious violation in the following \(X\) period of time?


## Early Warning

Using the same data set, facility owners or managers would pose the ML problem as an early warning problem. They'd like to know whether an inspector is going to visit their facility so they can prepare for it. They can define the problem like this:

> Will my facility be inspected in the next \(X\) period of time?

Note that in both cases, we are defining a period of time in which the event potentially will happen.


## What do they have in common?

For either problem, \(X\) could be a day, a week, month, a quarter, a year, 56 days, or some other time period.

Without going into detail, both problems use data where each row describes an **event** in which an **entity** was involved, and each event has a specific **outcome** or result.

The **entity** for both inspection prioritizations and early warnings in this tutorial is a food *facility*, and the **event** is an inspection. But the **outcome** differs: for inspections the outcome is *inspection failed* or *major violation found*, while for early warning the outcome is *inspected*.


## How do they differ?

Besides the obvious (i.e. the label), these ML's problem formulations have very different internal structure:

In the *EIS* problem **all** the entities of interest in a given period of time **have** a label. The *Inspections* problem does not have that luxury: from all the existing entities of interest only a bunch are *inspected* that means that only those inspected have a label (`True/False`) but all the remaining ones doesn't have one. This will be reflected, for example in the *training* matrices: you only train in the facilities that were inspected (so you will have less rows in them). Another impact will be in the metrics: you need to be very careful about interpreting the metrics in an inspections problem. Finally, when you are designing the field validation of your model, you need to take in account this selection bias, if not, you will be inspecting the same facilities over and over<sup><a id="fnr.6" class="footref" href="#fn.6">6</a></sup>

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> Several examples use this dataset, such as [City of Chicago Food Inspection Forecasting](https://chicago.github.io/food-inspections-evaluation/), [PyCon 2016 keynote: Built in Super Heroes](https://youtu.be/lyDLAutA88s), and [PyData 2016: Forecasting critical food violations at restaurants using open data](https://youtu.be/1dKonIT-Yak).

<sup><a id="fn.2" class="footnum" href="#fnr.2">2</a></sup> It is also possible to do "visit-level prediction" type of ML problem.

<sup><a id="fn.3" class="footnum" href="#fnr.3">3</a></sup> Examples include [Predictive Enforcement of Hazardous Waste Regulations](http://dsapp.uchicago.edu/projects/environment/) and [Targeting Proactive Inspections for Lead Hazards](http://dsapp.uchicago.edu/projects/health/lead-prevention/).

<sup><a id="fn.4" class="footnum" href="#fnr.4">4</a></sup> Examples include [Increasing High School Graduation Rates: Early Warnings and Predictive Systems](http://dsapp.uchicago.edu/projects/education/), [Building Data-Driven Early Intervention Systems for Police Officers](http://dsapp.uchicago.edu/projects/public-safety/police-eis/), and [Data-Driven Justice Initiative: Identifying Frequent Users of Multiple Public Systems for More Effective Early Assistance](http://dsapp.uchicago.edu/projects/criminal-justice/data-driven-justice-initiative/).

<sup><a id="fn.5" class="footnum" href="#fnr.5">5</a></sup> Defined as "bakery, banquet hall, candy store, caterer, coffee shop, day care center (for ages less than 2), day care center (for ages 2 – 6), day care center (combo, for ages less than 2 and 2 – 6 combined), gas station, Golden Diner, grocery store, hospital, long term care center(nursing home), liquor store, mobile food dispenser, restaurant, paleteria, school, shelter, tavern, social club, wholesaler, or Wrigley Field Rooftop" ([source](https://data.cityofchicago.org/api/views/4ijn-s7e5/files/O9cwLJ4wvxQJ2MirxkNzAUCCMQiM31DMzRkckMsKlxc?download=true&filename=foodinspections_description.pdf)).

<sup><a id="fn.6" class="footnum" href="#fnr.6">6</a></sup> This points is particularly acute: Imagine the scenario in which the *inspections* problem is **crime prediction** in order to send cops (inspectors)to that "risky" area (facilities)&#x2026;
