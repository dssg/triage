
**Source:** [triage/experiments/multicore.py#L0](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L0)



-------------------

### [insert_into_table](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L189)

```python
insert_into_table(insert_statements, feature_generator_factory, db_connection_string)
```





-------------------

### [build_matrix](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L205)

```python
build_matrix(build_tasks, planner_factory, db_connection_string)
```





-------------------

### [train_model](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L221)

```python
train_model(train_tasks, trainer_factory, db_connection_string)
```





-------------------

### [test_and_evaluate](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L238)

```python
test_and_evaluate(model_ids, predictor_factory, evaluator_factory, indiv_importance_factory, \
    test_store, db_connection_string, split_def, train_matrix_columns, config)
```






-------------------

## [MultiCoreExperiment](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L11)


The Base class for all Experiments.


#### MultiCoreExperiment.all_as_of_times
 
All 'as of times' in experiment config

Used for label and feature generation.

*Returns: (list) of datetimes*


#### MultiCoreExperiment.all_label_windows
 
All train and test label windows

*Returns: (list) label windows, in string form as they appeared in the experiment config*


#### MultiCoreExperiment.collate_aggregations
 
collate Aggregation objects used by this experiment.

*Returns: (list) of collate.Aggregation objects*


#### MultiCoreExperiment.feature_dicts
 
Feature dictionaries, representing the feature tables and columns configured in this experiment after computing feature groups.

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### MultiCoreExperiment.feature_table_tasks
 
All feature table query tasks specified by this Experiment

*Returns: (dict) keys are group table names, values are themselves dicts, each with keys for different stages of table creation (prepare, inserts, finalize) and with values being lists of SQL commands*


#### MultiCoreExperiment.full_matrix_definitions
 
Full matrix definitions

*Returns: (list) temporal and feature information for each matrix*


#### MultiCoreExperiment.master_feature_dictionary
 
All possible features found in the database. Not all features will necessarily end up in matrices

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### MultiCoreExperiment.matrix_build_tasks
 
Tasks for all matrices that need to be built as a part of this Experiment.

Each task contains arguments understood by Architect.build_matrix

*Returns: (list) of dicts*


#### MultiCoreExperiment.split_definitions
 
Temporal splits based on the experiment's configuration

*Returns: (dict) temporal splits*


*Example:*

```
{
  'beginning_of_time': {datetime},
  'modeling_start_time': {datetime},
  'modeling_end_time': {datetime},
  'train_matrix': {
  'matrix_start_time': {datetime},
  'matrix_end_time': {datetime},
  'as_of_times': [list of {datetime}s]
  },
  'test_matrices': [list of matrix defs similar to train_matrix]
}
```


-------------------

### [MultiCoreExperiment.`__init__`](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L12)

```python
__init__(self, n_processes=1, n_db_processes=1, *args, **kwargs)
```


Initialize self.  See help(type(self)) for accurate signature.





-------------------

### [MultiCoreExperiment.build_matrices](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L132)

```python
build_matrices(self)
```


Generate labels, features, and matrices

-------------------

### [MultiCoreExperiment.catwalk](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L21)

```python
catwalk(self)
```


Train, test, and evaluate models

-------------------

### [MultiCoreExperiment.parallelize](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L174)

```python
parallelize(self, partially_bound_function, tasks, n_processes, chunksize=1)
```




-------------------

### [MultiCoreExperiment.parallelize_with_success_count](https://github.com/dssg/triage/tree/master/triage/experiments/multicore.py#L107)

```python
parallelize_with_success_count(self, partially_bound_function, tasks, n_processes, chunksize=1)
```








