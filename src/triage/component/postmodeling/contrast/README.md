# Post-modeling Analysis

> What is the distribution of my predicted scores? What is generating a higher FPR in
> model x compared to model y? What is the most important predictor in my
> models?

This questions, and other ones, are the kind of inquiries that the `triage` user
may have in mind when exploring  models selected by the `Audition`
component. Choosing the right model for deployment and exploring its predictions
and behavior in time is a critical task. `postmodeling` will help to answer some
of these questions by exploring the outcomes and predictions of the model, and going "deeper"
into the model behavior across time and features. 

This library lies at the end of the `triage` pipeline and will use the output of
`Audition` and some of its selection rules as the input. The
`postmodeling_tutorial.ipynb` notebook contains a user guide with some questions
that the component is able to answer about the models, but the methods are
expandable and allow the user to develop new insights.  

## Configuration File

Before running the postmodeling notebook, the user must first define a series
of parameters relevant for the library: 

  - `project_path`: Triage's project path (same as your `config_file.yml`)
  - `audition_output_path`: path to Audition's output. If not passed, the class
    will use the listed `model_group_id`.
  - `thresholds`: list of thresholds. 
  - `baseline_query`: SQL query with baseline models metrics.
  - `max_depth_error_tree`: deep for error DTs.
  - other aesthetic arguments. 

This file is passed to the `PostmodelParameters` class to facilitate the use of
this parameters thorough the notebook.
[Here](triage/blob/master/examples/postmodeling_config_example.yaml) is an
example of a valid configuration file.  

## Two classes, two units of analysis: `model_id` and `model_group_id`

We can inspect models at two levels. The first, the individual model level
(identified by the `model_id`) with the `ModelEvaluator` class. The methods in
this class  will help us to answer questions about different relevant features
of each model: 

 - **Model Predictions**: 
     - `plot_score_label_distribution` shows a hued distribution plot of the predicted
       score colored by the label. A raw version of the plot without label can
       be plotted using the `plot_score_distribution` instead.

     - `plot_score_distribution_thresh` plots a score distribution plot with
       a dotted vertical line showing the "location" of a threshold defined by
       a model metric. For this the function need a pair of `param_type` and `param`
       (i.e. param_type='rank_abs', param=10).
       
 - **Feature relevance**:
     - `plot_feature_importances` shows an horizontal bar-plot with the top-n most
       relevant features defined by the model feature importance (when available). 

     - `plot_feature_importances_std_err` plots the feature importance of the
       top-n features with their standard deviation. This is highly informative
       in RandomForest models where each three can have different relevant
       features. The function needs the user to pass the project path to the `path`
       argument. More details in the [Configuration File](#configuration-file) section.

     - `plot_feature_group_average_importance` makes the same excercise, but it
       aggreagates (averages) feature importance metrics to the feature group
       level and plots the relevance of each feature group. 

 - **Model Matrix characteristics**: _These function need the user to pass
   Triage's project to the `path` argument. More details in the [Configuration
   File](#configuration-file) section._
   
     - `cluster_correlation_features` shows a correlation matrix ordered by the
         correlation between features. This plot can be subsetted by a feature
         group and explore the correlation in that set of the feature space. 
  
     - `cluster_correlation_sparsity` renders an image with the prediction matrix
       colored by their data availabilty. This plot can shows different
       consequences of data imputation and help the user to visualize the
       zero-only features (it happens with individual constant features). 
  
     - `plot_feature_distribution` plots the distribution of the top_n features
       comparing the positive and negative labeled entities. A three-column plot
       is rendered, the first two corresponding to the individual label plots,
       and the third one corresponding to both. 

 - **Model Metrics**:
   - `plot_ROC` plots the AUC-ROC curve plot for the selected model. This is
     just a wrapper of the `sklearn` library function to be compatible with
     triage.

   - `plot_precision_recall_n` renders the precision/recall curves for the
     desired model
     
   - `plot_recall_fpr_n` plots the false positive rate against the threshold of
     absolute population (another way of exploring recall).

 - **Error Analysis**: Error analysis is a powerful tool to identify where the
   model is making classification mistakes. Error analysis involves several
   other hidden functions, but it can be completely run using the
   `error_analysis` function. This function will label the errors and use
   a `DecisionTreeClassifier` to classify the errors and print the modeled
   tree. This function needs a parameters object defined in the configuration
   file. More details in the [Configuration File](#configuration-file)
   section.

 - **Crosstabs Analysis**: `crosstabs_ratio_plot` will plots the mean ratio for
   all relevant features in the model. This allows a good comparison of
   true/false predicted groups and get the key differences between groups in
   the feature space. The function expects that the user run
   [crosstabs](docs/sources/postmodeling/index.md#crosstabs)
   first. 

The `ModelEvaluator` also contain a set of miscellaneous methods to retrieve
different model results: 

|            method           |                                     Description                                    |
|:---------------------------:|:----------------------------------------------------------------------------------:|
|          `metadata`         | Table with model metadata elements. You can explore the class `.__dict__`  as well |
|        `predictions`        | Returns a `pd.DataFrame` with model predictions                                    |
|    `feature_importances`    | Returns a `pd.DataFrame` with model feature importances                            |
| `feature_group_importances` | Returns a `pd.DataFrame` with the importances mean  per feature group              |
|        `test_metrics`       | Returns a `pd.DataFrame` with test model evaluation metrics                        |
|       `train_metrics`       | Returns a `pd.DataFrame` with train model evaluation metrics                       |
|         `crosstabs`         | Returns a `pd.DataFrame` with model cross tabs (when available)                    |
|        `preds_matrix`       | Returns a `pd.DataFrame` with the test matrix                                      |
|        `train_matrix`       | Returns a `pd.DataFrame` with the train matrix                                     |

The second method will help us to answer questions related with models across
time. The `model_group_id` contains all equal specified models with different
`as_of_dates` (test/validation dates). The `ModelGroupEvaluator`
contains modules helpful for:

 - **Models behavior across time**:
     - `plot_prec_across_time`: this function will render the metric behavior
       across time. This metric can be any of the common used metric on the
       triage evaluations (i.e. `precision`, `recall`, etc.). The function can
       also take an user predefined baseline, which is defined in the
       postmodeling configuration file.

     - `feature_loi_loo`: will renders the same plot as the
       `plot_prec_across_time`, but it will detect the model features and
       identify if the included models are part of a LOO (leave-one-out) or LOI
       (leave-one-in) experiment and label the models and they LOI or LOO
       feature accordingly.

 - *Model group predictions and feature comparisons*: These functions listed
   below allow the user to compare model predictions and feature importances
   using different methods. Although, be aware that some comparisons between
   individual models inside the model groups can be invalid (i.e. comparing the
   `model_id = 1` of the `model_group_id = 1` with an `as_of_date = 2012-01-01`
   with  the `model_id = 5` of the `model_group_id = 2` with an `as_of_date
   = 2013-01-01`). For this reason all the functions listed include
   a `temporal_comparison=False` argument. This option will group the models by
   `as_of_date` and make comparisons only between same-prediction-time models. 
 
     - `plot_ranked_correlation_preds`: this function will plot a heatmap with
        the predictions ranked correlation for each of the individual `model_group_id`
        and its `model_id` (which are equal but with different test_date). This
        function uses the hidden function `_rank_corr_df` which calculates the
        correlation in a pairwise manner. For succesfuly compare predictions,
        the user must especify a threshold metric (`param_type` and `param`) to
        select the true positive predictions. 

     - `plot_ranked_correlation_features`: just as the `_preds` function above,
       this function will plot a heatmap with the rank correlation matrix for
       each of the model groups (you can pass a `top_n_features` argument to
       select the top-n features to compare). 

      - `plot_jaccard_preds`: as the rank correlation functions, this function
        compare the prediction set overlap between different model groups. This
        function needs the user to pass a `param_type` and a `param` value to
        select the true positive entities to compare.

      - `plot_jaccard_features`: Jaccard overlap for feature importances. Just
        as the above function, the function will plot the feature importance set
        overlap for each model inside the model group. 

      - `plot_preds_comparison`: plots the comparison of score distributions
        across models in the model groups. This plot needs a `param_type` and
        a `param` to define a threshold to define the false positives. 

The `ModelGroupEvaluator` class also has different objects that can be useful
for further analysis:

|       method       | Description                                                                                                     |
|:------------------:|-----------------------------------------------------------------------------------------------------------------|
|     `metadata`     | Table with model group elements. You can explore the class `.__dict__`  as well                                 |
|     `model_id`     | List of `model_id` inside each model group.                                                                     |
|    `model_hash`    | List of `model_hash` inside each model group.                                                                   |
|  `hyperparameters` | List of model group hyperparameters.                                                                            |
|  `train_end_time`  | List of model group train end times.                                                                            |
|    `model_type`    | List of model types for each model group.                                                                       |
|    `predictions`   | Returns a `pd.DataFrame` with model predictions.                                                                |
|  `feature_groups`  | Returns a `pd.DataFrame` with model feature groups  used by each model-group (also experiment type: LOI or LOO) |
| `same_time_models` | Returns a `pd.DataFrame` with `as_of_dates` and  a list of models estimated in that date.                       | 
