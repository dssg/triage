
**Source:** [triage/experiments/base.py#L0](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L0)

**Global Variables**
---------------
- **CONFIG_VERSION**

-------------------

### [dt_from_str](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L25)

```python
dt_from_str(dt_str)
```






-------------------

## [ExperimentBase](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L29)


The Base class for all Experiments.


#### ExperimentBase.all_as_of_times
 
All 'as of times' in experiment config

Used for label and feature generation.

*Returns: (list) of datetimes*


#### ExperimentBase.all_label_windows
 
All train and test label windows

*Returns: (list) label windows, in string form as they appeared in the experiment config*


#### ExperimentBase.collate_aggregations
 
collate Aggregation objects used by this experiment.

*Returns: (list) of collate.Aggregation objects*


#### ExperimentBase.feature_dicts
 
Feature dictionaries, representing the feature tables and columns configured in this experiment after computing feature groups.

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### ExperimentBase.feature_table_tasks
 
All feature table query tasks specified by this Experiment

*Returns: (dict) keys are group table names, values are themselves dicts, each with keys for different stages of table creation (prepare, inserts, finalize) and with values being lists of SQL commands*


#### ExperimentBase.full_matrix_definitions
 
Full matrix definitions

*Returns: (list) temporal and feature information for each matrix*


#### ExperimentBase.master_feature_dictionary
 
All possible features found in the database. Not all features will necessarily end up in matrices

*Returns: (list) of dicts, keys being feature table names and values being lists of feature names*


#### ExperimentBase.matrix_build_tasks
 
Tasks for all matrices that need to be built as a part of this Experiment.

Each task contains arguments understood by Architect.build_matrix

*Returns: (list) of dicts*


#### ExperimentBase.split_definitions
 
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

### [ExperimentBase.`__init__`](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L33)

```python
__init__(self, config, db_engine, model_storage_class=None, project_path=None, replace=True)
```


Initialize self.  See help(type(self)) for accurate signature.





-------------------

### [ExperimentBase.build_matrices](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L419)

```python
build_matrices(self)
```


Generate labels, features, and matrices

-------------------

### [ExperimentBase.catwalk](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L424)

```python
catwalk(self)
```


Train, test, and evaluate models

-------------------

### [ExperimentBase.generate_labels](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L369)

```python
generate_labels(self)
```


Generate labels based on experiment configuration

Results are stored in the database, not returned

-------------------

### [ExperimentBase.generate_sparse_states](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L380)

```python
generate_sparse_states(self)
```




-------------------

### [ExperimentBase.initialize_components](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L186)

```python
initialize_components(self)
```




-------------------

### [ExperimentBase.initialize_factories](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L85)

```python
initialize_factories(self)
```




-------------------

### [ExperimentBase.log_split](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L392)

```python
log_split(self, split_num, split)
```




-------------------

### [ExperimentBase.matrix_store](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L401)

```python
matrix_store(self, matrix_uuid)
```


Construct a matrix store for a given matrix uuid, using the Experiment's #matrix_store_class

*Args:*

  matrix_uuid (string) A uuid for a matrix

-------------------

### [ExperimentBase.run](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L429)

```python
run(self)
```




-------------------

### [ExperimentBase.update_split_definitions](https://github.com/dssg/triage/tree/master/triage/experiments/base.py#L385)

```python
update_split_definitions(self, new_split_definitions)
```


Update split definitions

*Args: (dict) split definitions (should have matrix uuids)*






