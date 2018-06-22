# Feature Generation Recipe Book

This document is a collection of 'collate' aggregate features that we have found useful to create in Triage that may not be apparent at first.

For an introduction to feature generation in Triage, refer to [Dirty Duck Feature Generation](https://dssg.github.io/dirtyduck/#orgaae2e66)

## Age

Calculating the age of a person in a collate feature can be calculated if you know the date of birth with the use of the `collate_date` special variable. This variable is marked as a placeholder in the feature quantity input, but is replaced with each as-of-date when features are being calculated. Combined with the Postgres `age` function, you can calculate a person's age as a feature.

For this example, let's assume you have a column called 'dob' that is a timestamp (or anything that can be cast to a date) in your source table. The `feature_aggregation`'s quantity would be: 

```EXTRACT(YEAR FROM AGE('{collate_date}'::DATE, dob::DATE))```

If Triage is calculating this for the as-of-date '2016-01-01', it will internally expand the `collate_date` out to:
```EXTRACT(YEAR FROM AGE('2016-01-01'::DATE, dob::DATE))```

In context, a feature aggregate that uses age may look more like:

```
    aggregates:
      - # age in years 
        quantity:
          age: "EXTRACT(YEAR FROM AGE('{collate_date}'::DATE, dob::DATE))"
        metrics: ['max']
```


Here, we call the feature 'age' and since everything in collate is defined as an aggregate, we pick 'max'; Any records for the same person and as-of-date should have the same 'dob', so the choice of aggregate metric here should not matter, but in these cases 'max' is the standard aggregate metric of choice in Triage.
