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

 - **Error Analysis**: Error analysis is a powerful tool to identify where the
   model is making classification mistakes. Error analysis involves several
   other hidden functions, but it can be completely run using the
   `error_analysis` function. This function will label the errors and use
   a `DecisionTreeClassifier` to classify the errors and print the modeled tree.

 - **Model Metrics**:
   - 

about model predictions, feature relevance, prediction matrix characteristics,
prediction error analysis, and `sklearn` classifier object metrics. The
`ModelEvaluator` class contains different methods with the aforementioned
questions. 


We can analyze models by getting its individual metrics or comparing between
them. The first step would include a set of basic metric for each model, such
as:
 - [x] Raw score distributions: histogram of the scores
 - [x] Score distributions: compare the distributions of labels in the test
   matrix across the predicted score.
 - [x] Feature importances: 
 
 ...Where possible, `triage` will calculate the general feature importances for
 each model (for Random Forest and other decision trees, this process is made
 extracting the `_feature_importances` from `sklearn`). Nonetheless, when
 available we can also user `feature_groups` as a way to improve the
 interpretability of the feature importances. 
 
 - [x] Feature importances using standard deviations, and other ways to get
   these. 
 - [x] Matrix metrics: basic descriptives of the test matrix (n, labels, etc.)
 - [x] Model metrics: ROC Curve, Precision vs. Recall, Recall and FDR vs.
   Depth
 - [x] Error trees with different labels (i.e. complete residuals, FPR, and
   FNR).
 
The second step comprises different ways of comparing models: 
* List comparisons: Compare list generated by each model:
 - [x] Jaccard Similarity over *top_k* predictions and features
 - [x] Overall rank correlation over *top_k* predictions and features
 - [x] Rank Correlation where we can label [0 vs. 1]
 - [x] KL divergence to compare distributions in feature importances

Postmodeling also involves the correction of scores and a more in-depth analysis
of the selected models. Some of this tasks can be also achieved using this
library, like:

 - [ ] Probability Calibration: probability calculations for score deciles
 - [ ] Crosstabs: t-test comparing feature values between classified groups.
 - [ ] Bias report: *Aequitas* 

A [notebook](https://github.com/dssg/triage/blob/cli_postmodeling/src/triage/component/postmodeling/contrast/postmodeling_tutorial.ipynb)
is included with a tutorial of the library using the [Dirty
Duck](https://github.com/dssg/dirtyduck) data. 

