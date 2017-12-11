
**Source:** [triage/experiments/singlethreaded.py#L0](https://github.com/dssg/triage/tree/master/triage/experiments/singlethreaded.py#L0)





-------------------

## [SingleThreadedExperiment](https://github.com/dssg/triage/tree/master/triage/experiments/singlethreaded.py#L6)


The Base class for all Experiments.


#### SingleThreadedExperiment.all_as_of_times
 
All 'as of times' in experiment config

Used for label and feature generation.

*Returns: (list) of datetimes*


#### SingleThreadedExperiment.all_label_windows
 
All train and test label windows

*Returns: (list) label windows, in string form as they appeared in the experiment config*


#### SingleThreadedExperiment.collate_aggregations
 
collate Aggregation objects used by this experiment.

*Returns: (list) of collate.Aggregation objects*


#### SingleThreadedExperiment.feature_dicts
 
Feature dictionaries, representing the feature tables and columns configured in this experiment after computing feature groups.

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### SingleThreadedExperiment.feature_table_tasks
 
All feature table query tasks specified by this Experiment

*Returns: (dict) keys are group table names, values are themselves dicts, each with keys for different stages of table creation (prepare, inserts, finalize) and with values being lists of SQL commands*


#### SingleThreadedExperiment.full_matrix_definitions
 
Full matrix definitions

*Returns: (list) temporal and feature information for each matrix*


#### SingleThreadedExperiment.master_feature_dictionary
 
All possible features found in the database. Not all features will necessarily end up in matrices

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### SingleThreadedExperiment.matrix_build_tasks
 
Tasks for all matrices that need to be built as a part of this Experiment.

Each task contains arguments understood by Architect.build_matrix

*Returns: (list) of dicts*


#### SingleThreadedExperiment.split_definitions
 
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

### [SingleThreadedExperiment.`__init__`](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L33)

```python
__init__(self, config, db_engine, model_storage_class=None, project_path=None, replace=True)
```


Initialize self.  See help(type(self)) for accurate signature.





-------------------

### [SingleThreadedExperiment.build_matrices](https://github.com/dssg/triage/tree/master/triage/experiments/singlethreaded.py#L7)

```python
build_matrices(self)
```


Generate labels, features, and matrices

-------------------

### [SingleThreadedExperiment.catwalk](https://github.com/dssg/triage/tree/master/triage/experiments/singlethreaded.py#L17)

```python
catwalk(self)
```


Train, test, and evaluate models





