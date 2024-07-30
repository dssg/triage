# Postmodeling

This module deals with the analyses we perform once the model building is finished. While the individual analysis pieces are valid for any ML model that is built, the current implementation of this module is tightly coupled to the triage infrasturcute. 

## Postmodeling Report

### Generating a Experiment Summary Report Automatically
This part of the modeling processes is often interactive. As a starting point to this, a "postmodeling report" can be automatically generated right after the completion of the triage experiment, by performing the following step.  

1. Copy the triage_experiment_report_template.ipynb to your project repo

2. Install `nbcovert` to the same virtual envrionment where triage is installed for your project
```
$ pip install nbconvert
```

3. Copy the following code snippet to your `run.py` (or other `.py` script that calls triage) if you are using the Python interface for triage (currently the automatic report generation is not supported in the triage CLI version).  

```
def generate_experiment_report():

    # Path to where you save the notebook template in 
    template_path = '/path/to/the/notebook/template'

    # Specify where you will save the executed notebook (recommend not to overwrite the template)
    output_path = 'path/to/the/notebook/output'

    shutil.copyfile(template_path, output_path)

    os.system(f'jupyter nbconvert --execute --to notebook {output_path}')
    os.system(f'jupyter nbconvert --to html {output_path}')
```

4. Call the `generate_experiment_report()` function after the `.run()` function of the `SingleThreadedExperiment` or `MultiCoreExperiment` class.


Once the script is executed, a notebook will be generated that overviews the following:
- Temporal validation splits
- Model types and objects that were built
- Predictors/features that were built and some stats on missingness of each feature
- Performance of all models wrt to a priority metric that was specified in the config (defaults to recall@1_pct)
- If a bias audit was specified, how well the models are performing interms of bias wrt to a prioritized bias metric and cohort groups (e.g., protected demographic groups) specified in the experiment config (The metric defaults to `tpr_disparity` and by default shows metric performance for all subgroups in the attributes specfied in the`bias_audit`)

To specify these metrics, you can add the following keys to the experiment configuration `.yaml` file. 

```
scoring:
    # Append these key-value pairs to the scoring section
    priority_metric: 'precision@'
    priority_parameter: '100_pct' 
  
bias_audit:
    ## Append these key-value pairs to the bias_audit section (if a bias audit is performed)
    priority_metric: 'tpr_disparity'

  priority_groups:
    'race':
      - 'Black/African American'
    'gender':
      - '2~Female'
```

In addition, for the best model of each model type (e.g., best Random Forest, best Decision Tree) based on the prioritized performance metric, it shows feature importance values, recall curve comparisons, feature importance comparisons across model pairs. 

5. Once the code is run, the notebook will be generated as an `.html` and an `.ipynb`. The `html` can serve as a report, and the `.ipynb` can serve as a starting point to the further postmodeling analysis.  

### Interactive Notebook

If you choose not to run the modeling report automatically with the triage experiment or you need to generate the report for an older experiment, you can update the following parameters at the top of the notebook. 

```
# The most recent completed experiment hash
experiment_hashes = [get_most_recent_experiment_hash(db_engine)]

# Model Performance metric and threshold defaulting to reacll@1_pct
performance_metric = 'recall@'
threshold = '1_pct'

# Bias metric defaults to tpr_disparity and bias metric values for all groups generated (if bias audit specified in the experiment config)
bias_metric = 'tpr_disparity'
bias_priority_groups=None

```


## Functionality that we support
 - for each model group and time (each model id), show:
    - PR-k curve
    - feature importance
    - bias
    - list
       - top k list
       - list cross-tab
       - list descriptives
    - error analysis
 - for each model group, compare
    - feature importances
    - list
    - performance
    - bias

### Sample notebook to use
- postmodeling_report_example_acdhs_housing.ipynb
- places to change settings


### Post triage run report generation
- report_generator.py
- instructions on how to run it
- sample report output
