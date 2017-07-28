=======
collate
=======


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


* Free software for noncommercial use: `UChicago open source license <https://github.com/dssg/collate/blob/master/LICENSE>`_.
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

    SELECT license_no, sum((results = 'Fail')::int) as failed_sum
    FROM food_inspections GROUP BY license_no;

In collate, this aggregated column would be defined as::

    Aggregate({"failed": "(results = 'Fail')::int"}, "sum")

Note that the SQL query is split into two parts: the first argument to ``Aggregate``
is the computation to be performed and gives it a name (as a dictionary key), and
the second argument is the reduction function to perform. Splitting the SQL like
this makes it easy to generate lots of composable features as the outer product
of these two lists.  For example, you may also be interested in the proportion
of inspections that resulted in a failure in addition to the total number. This is
easy to specify with the average value of the `failed` computation::

    Aggregate({"failed": "(results = 'Fail')::int"}, ["sum","avg"])


Aggregations in collate easily aggregate this single feature across different spatiotemporal groups, e.g.::

    Aggregate({"failed": "(results = 'Fail')::int"}, ["sum","avg"])
    st = SpacetimeAggregation([fail],
	                           from_obj='food_inspections',
                               groups=['license_no','zip'],
                               intervals={"license_no":["2 year", "3 year"], "zip": ["1 year"]},
                               dates=["2016-01-01", "2015-01-01"],
                               date_column="inspection_date",
                               schema='test_collate')

The ``SpacetimeAggregation`` object encapsulates the ``FROM`` section of the query
(in this case it's simply the inspections table), as well as the ``GROUP BY``
columns.  Not only will this create information about the individual restaurants
(grouping by ``license_no``), it also creates "neighborhood" columns that add
information about the region in which the restaurant is operating (by grouping by
``zip``).

Even more powerful is the sophisticated date range partitioning that the
``SpacetimeAggregation`` object provides.  It will create multiple queries in
order to create the summary statistics over the past 1, 2, or 3 years, looking
back from either Jan 1, 2015 or Jan 1 2016. Executing this set of queries with::

    st.execute(engine.connect()) # with a SQLAlchemy engine object

will create three new tables in the ``test_collate`` schema. The table
``food_inspections_license_no`` will contain four feature columns for each
license that describe the total number and proportion of failures over the past
two or three years, with a date column that states whether it was looking
before 2016 or 2015. Similarly, a ``food_inspections_zip`` table will have two
feature columns for every zip code in the database, looking at the total and
average number of failures in that neighborhood over the year prior to the date
in the date column. Finally, the ``food_inspections_aggregate`` table joins
these results together to make it easier to look at both neighborhood and
restaurant-level effects for any given restaurant.

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
~~~~~~~~~~~~~~~~~~~~~~~~~
TODO

Technical details
=================
