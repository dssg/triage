===============================
collate
===============================


.. image:: https://img.shields.io/pypi/v/collate.svg
        :target: https://pypi.python.org/pypi/collate

.. image:: https://travis-ci.org/dssg/collate.svg?branch=master
        :target: https://travis-ci.org/dssg/collate

.. image:: https://readthedocs.org/projects/collate/badge/?version=latest
        :target: https://collate.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/dssg/collate/shield.svg
     :target: https://pyup.io/repos/github/dssg/collate/
     :alt: Updates

.. image:: https://codecov.io/gh/dssg/collate/branch/master/graph/badge.svg
	 :target: https://codecov.io/gh/dssg/collate
	 :alt: Code Coverage


Aggregated feature generation made easy.


* Free software: MIT license
* Documentation: https://collate.readthedocs.io.

Overview
========

Collate allows you to easily specify and execute statements like “find the number of restaurants in a given zip code that have had food safety violations within the past year.”  The real power is that it allows you to vary both the spatial and temporal windows, choosing not just zip code and one year, but a range over multiple partitions and times. Specifying features is also easier and more efficient than writing raw sql. Collate will automatically generate and execute all the required SQL scripts to aggregate the data across many groups in an efficient manner. We mainly use the results as features in machine learning models.

Inputs
======

Take for example `food inspections data from the City of Chicago <https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5>`_. The table looks like this:


============= =========== ===== =============== ========== =========== ===
inspection_id license_no  zip   inspection_date results    violations  ...
============= =========== ===== =============== ========== =========== ===                                                                                 
1966765       80273       60636 2016-10-18      No Entry               ...
1966314       2092894     60640 2016-10-11      Pass       …CORRECTED… ...
1966286       2215628     60661 2016-10-11      Pass w/ C… …HAZARDOUS… ...
1966220       2424039     60620 2016-10-07      Pass                   ...
============= =========== ===== =============== ========== =========== ===                                                                                 

There are two spatial levels in the data: the specific restaurant (by its license number) and the zip code. And there is a date.

An example of an aggregate feature is the number of failed inspections. In raw SQL this could be calculated, for each restaurant, as so::

    SELECT license, sum((results = 'Fail')::int) as fail_count
    FROM food_inspections;
	
In collate, this aggregate would be defined as::

	Aggregate({"fail_count": "(results = Fail)::int"}, "sum")


Aggregations in collate easily aggregate this single feature across different spatiotemporal groups, e.g.::

    fail = Aggregate({"fail_count": "(results = Fail)::int"}, "sum")
    st = SpacetimeAggregation([fail], 'food_inspections',
                               group_intervals={"license_no":["2 year", "3 year"], "zip": ["1 year"]},
                               dates=["2016-01-01", "2015-01-01"],
                               date_column="inspection_date")


will aggregate this feat


Another advantage of collate is quickly defining many aggregations. For example to add another feature which is the proportion of inspections which failed in the given group we can simply pass a list of functions to Aggregate.

::

    Aggregate({"fail_count": "(results = Fail)::int"}, ["sum", "avg"])


Outputs
=======

The main output of a collate aggregation is a database table with all of the aggregated features joined to a list of entities.


TODO: sample rows from the above aggregation.


Usage Examples
==============

Multiple quantities
~~~~~~~~~~~~~~~~~~~
TODO

Multiple functions
~~~~~~~~~~~~~~~~~~
TODO

Tuple quantity
~~~~~~~~~~~~~~
TODO

Date substitution
~~~~~~~~~~~~~~~~~
TODO

Categorical counts
~~~~~~~~~~~~~~~~~~
TODO

Naming of features
~~~~~~~~~~~~~~~~~~
TODO

More complicated from_obj
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TODO

Technical details
=================
