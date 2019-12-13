# Using Triage for a Project: Workflow Tips


## Step 1: Get your data set up
Triage needs data in a postgres database, with at least one table that contains `events` (one per row) and 
`entities` of interest (people, place, organization, etc.), a `timestamp` (specifyinfg when the event occurred), and 
additional attributes of interest about the event and/or entity (demographics for example).

We will need a database credential file [sample] that contains the name of the database, server, username, and password to use to connect to it. 

## Iteration 1: Quick Check

This set up will run a quick sanity check to make sure everything is set up correctly and that triage runs with your data and set up.

### Configuration
The configuration file has a lot of sections. In the first iteration, we'll set up the minimal parameters necessary to get started.

1. Define `outcome/label` of interest: This is a SQL query that defines outcome we want to predict and needs to return The query must return two columns: 
entity_id and outcome, based on a given as_of_date and label_timespan. 
See our [guide to Labels](https://dssg.github.io/triage/experiments/cohort-labels/)
2. Define `training_label_
timespan` ( = '12month' for example for predict one year out)
3. Specify `features` (at least one is neded to run Triage)
4. Specify `evaluation metrics` to calculate (at least one is needed)
5. Specify `model grid` for Triage to run models with different hyperparameters (set to quickstart at this time)

### Run Triage

1. Check triage config 
2. Run the pipeline

### Explore Results

1. Audtion
2. Post-Modeling

## Iteration 2: Refine the cohort and temporal setup

## Iteration 3: Add more data/features, models and hyperparameters, and evaluation metrics of interest

## Iteration 4: Explore additional labels/outcomes and calculation evaluation metrics on subsets of entities that may be of special interest
