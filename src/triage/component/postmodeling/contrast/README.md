# Post-modeling Analysis

> ¿What is the distribution of my scores? ¿What is generating a higher FPR in
> model x compared to model y? ¿What is the single most important feature in my
> models?

This questions, and other ones, are the kind of inquiries that the `triage`user
may have in mind when scrolling trough the models selected by the `Audition`
component. Choosing the right model for deployment and exploring its predictions
and behavior in time is a pivotal task. `postemodeling` will help to answer some
of this questions by exploring the outcomes of the model, and exploring "deeply"
into the model behavior across time and features. 

This library lays at the end of the `triage` pipeline and will use the output of
`Audition` and some of its selection rules as a main input. The
`postmodeling_tutorial.ipynb` notebook contains a user guide with some questions
that the component is able to answer about the models, but the methods are
expandable and allow the user to develop new insights.  

Here is a list of general functions that helps the `triage` user to understand
and analyze selected models:

## Two classes, two units of analysis: `model_id` and `model_group_id`

We can inspect models in two levels. The first, the individual model level
(identified by the `model_id`) with the `ModelEvaluator` class. The methods in
this class  will help us to answer questions about different relevant features
of each model: 

 - **Model Predictions**: 
     - `plot_score_label_distribution` shows a hued distribution plot of the predicted
       score colored by the label. A raw version of the plot without label can
       be plotted using the `plot_score_distribution` instead.

     - `plot_score_distribution_thresh` plots a score distribution plot with
       a dotted vertical line showing the "location" of a threshold defined by
       a model metric (i.e. precision@pct10).
       
 - **Feature relevance**:
     - `plot_feature_importances` shows an horizontal bar-plot with the top-n most
       relevant features defined by the model feature importance (when available). 

     - `plot_feature_importances_std_err` plots the feature importance of the
       top-n features with their standard deviation. This is highly informative
       in RandomForest models where each three can have different relevant
       features. 

     - `plot_feature_group_average_importance` makes the same excersice, but it
       aggreagates (averages) feature importance metrics to the feature group
       level and plots the relevance of each feature grup. 

 - **Model Matrix characteristics**:
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
   tree.

 - **Crosstabs Analysis**: `crosstabs_ratio_plot` will plots the mean ratio for
   all relevant features in the model. This allows a good comparison of
   true/false predicted groups and get the key differences between groups in the
   feature space.

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
`as_of_dates` (test/validation dates) per entity. The `ModelGroupEvaluator`
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

 - *Model group comparisons*

