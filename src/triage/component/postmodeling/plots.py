# coding: utf-8

def plot_ROC(self,
             figsize=(16, 12),
             fontsize=20):
    '''
    Plot an ROC curve for this model and label it with AUC
    using the sklearn.metrics methods.
    Arguments:
        - figsize (tuple): figure size to pass to matplotlib
        - fontsize (int): fontsize for title. Default is 20.
    '''

    fpr, tpr, thresholds, auc = self.compute_AUC()
    auc = "%.2f" % auc

    title = 'ROC Curve, AUC = ' + str(auc)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, "#000099", label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(loc='lower right')
    plt.title(title, fontsize=fontsize)
    plt.show()

def plot_recall_fpr_n(self,
                      figsize=(16, 12),
                      fontsize=20):
    '''
    Plot recall and the false positive rate against depth into the list
    (esentially just a deconstructed ROC curve) along with optimal bounds.
    Arguments:
        - figsize (tuple): define a plot size to pass to matplotlib
        - fontsize (int): define a custom font size to labels and axes
    '''


    _labels = self.predictions.filter(items=['label_value', 'score'])

    # since tpr and recall are different names for the same metric, we can just
    # grab fpr and tpr from the sklearn function for ROC
    fpr, recall, thresholds, auc = self.compute_AUC()

    # turn the thresholds into percent of the list traversed from top to bottom
    pct_above_per_thresh = []
    num_scored = float(len(_labels.score))
    for value in thresholds:
        num_above_thresh = len(_labels.loc[_labels.score >= value, 'score'])
        pct_above_thresh = num_above_thresh / num_scored
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    # plot the false positive rate, along with a dashed line showing the optimal bounds
    # given the proportion of positive labels
    plt.clf()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot([_labels.label_value.mean(), 1], [0, 1], '--', color='gray')
    ax1.plot([0, _labels.label_value.mean()], [0, 0], '--', color='gray')
    ax1.plot(pct_above_per_thresh, fpr, "#000099")
    plt.title('Deconstructed ROC Curve\nRecall and FPR vs. Depth',fontsize=fontsize)
    ax1.set_xlabel('Proportion of Population', fontsize=fontsize)
    ax1.set_ylabel('False Positive Rate', color="#000099", fontsize=fontsize)
    plt.ylim([0.0, 1.05])


def plot_precision_recall_n(self,
                            figsize=(16, 12),
                            fontsize=20):
    """
    Plot recall and precision curves against depth into the list.
    """


    _labels = self.predictions.filter(items=['label_value', 'score'])

    y_score = _labels.score
    precision_curve, recall_curve, pr_thresholds = \
        metrics.precision_recall_curve(_labels.label_value, y_score)

    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]

    # a little bit of a hack, but ensure we start with a cut-off of 0
    # to extend lines to full [0,1] range of the graph
    pr_thresholds = np.insert(pr_thresholds, 0, 0)
    precision_curve = np.insert(precision_curve, 0, precision_curve[0])
    recall_curve = np.insert(recall_curve, 0, recall_curve[0])

    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(pct_above_per_thresh, precision_curve, "#000099")
    ax1.set_xlabel('Proportion of population', fontsize=fontsize)
    ax1.set_ylabel('Precision', color="#000099", fontsize=fontsize)
    plt.ylim([0.0, 1.05])
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, "#CC0000")
    ax2.set_ylabel('Recall', color="#CC0000", fontsize=fontsize)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Precision-recall at x-proportion", fontsize=fontsize)
    plt.show()


def plot_prec_across_time(self,
                          param_type=None,
                          param=None,
                          metric=None,
                          baseline=False,
                          baseline_query=None,
                          df=False,
                          figsize=(12, 16),
                          fontsize=20):

    '''
    Plot precision across time for all model_group_ids, and baseline,
    if available.

    This function plots the performance of each model_group_id following an
    user defined performance metric. First, this function check if the
    performance metrics are available to both the models, and the baseline.
    Second, filter the data of interest, and lastly, plot the results as
    timelines (model_id date).

    Arguments:
        - param_type (string): A parameter string with a threshold
        definition: rank_pct, or rank_abs. These are usually defined in the
        postmodeling configuration file.
        - param (int): A threshold value compatible with the param_type.
        This value is also defined in the postmodeling configuration file.
        - metric (string): A string defnining the type of metric use to
        evaluate the model, this can be 'precision@', or 'recall@'.
        - baseline (bool): should we include a baseline for comparison?
        - baseline_query (str): a SQL query that returns the evaluation data from
        the baseline models. This value can also be defined in the
        configuration file.
        - df (bool): If True, no plot is rendered, but a pandas DataFrame
        is returned with the data.
        - figsize (tuple): tuple with figure size parameters
        - fontsize (int): Fontsize for titles
    '''

    # Load metrics and prepare data for analysis
    model_metrics = self.metrics
    model_metrics[['param', 'param_type']] = \
            model_metrics['parameter'].str.split('_', 1, expand=True)
    model_metrics['param'] =  model_metrics['param'].astype(str).astype(float)
    model_metrics['param_type'] = model_metrics['param_type'].apply(lambda x: 'rank_'+x)

    # Filter model_group_id metrics and create pivot table by each
    # model_group_id.
    model_metrics_filter = model_metrics[(model_metrics['metric'] == metric) &
                                  (model_metrics['param'] == param) &
                                  (model_metrics['param_type'] == param_type)].\
            filter(['model_group_id', 'model_id', 'as_of_date_year',
                    'value'])

    if baseline == True:

        baseline_metrics = pd.read_sql(baseline_query, con=self.engine)
        baseline_metrics[['param', 'param_type']] = \
                baseline_metrics['parameter'].str.split('_', 1, expand=True)
        baseline_metrics['param'] = baseline_metrics['param'].astype(str).astype(float)
        baseline_metrics['param_type'] = baseline_metrics['param_type'].\
                apply(lambda x: 'rank_'+x)

        # Filter baseline metrics and create pivot table to join with
        # selected models metrics
        baseline_metrics_filter =  baseline_metrics.\
                filter(['model_group_id', 'model_id', 'as_of_date_year', 'value'])

        baseline_metrics_filter['model_group_id'] = \
                baseline_metrics_filter.model_group_id.\
                apply(lambda x: 'baseline_' + str(x))

        # Join tables by index(as_of_date_year)
        model_metrics_filter = \
        model_metrics_filter.append(baseline_metrics_filter, sort=True)

    model_metrics_filter['as_of_date_year'] = \
            model_metrics_filter.as_of_date_year.astype('int')
    model_metrics_filter = model_metrics_filter.sort_values('as_of_date_year')

    if df == True:
        return model_metrics_filter

    else:
        try:
            sns.set_style('whitegrid')
            fig, ax = plt.subplots(figsize=figsize)
            for model_group, df in model_metrics_filter.groupby(['model_group_id']):
                ax = ax = df.plot(ax=ax, kind='line',
                                  x='as_of_date_year',
                                  y='value',
                                  marker='d',
                                  label=model_group)
            plt.title(str(metric).capitalize() +\
                      ' for selected model_groups in time.',
                      fontsize=fontsize)
            ax.tick_params(labelsize=16)
            ax.set_xlabel('Year of prediction (as_of_date)', fontsize=20)
            ax.set_ylabel(f'{str(metric)+str(param_type)+str(param)}',
                          fontsize=20)
            plt.xticks(model_metrics_filter.as_of_date_year.unique())
            plt.yticks(np.arange(0,1,0.1))
            legend=plt.legend(bbox_to_anchor=(1.05, 1),
                       loc=2,
                       borderaxespad=0.,
                       title='Model Group',
                       fontsize=fontsize)
            legend.get_title().set_fontsize('16')

        except TypeError:
                print(f'''
                      Oops! model_metrics_pivot table is empty. Several problems
                      can be creating this error:
                      1. Check that {param_type}@{param} exists in the evaluations
                      table
                      2. Check that the metric {metric} is available to the
                      specified {param_type}@{param}.
                      3. You basline model can have different specifications.
                      Check those!
                      4. Check overlap between baseline dates and model dates.
                      The join is using the dates for doing these, and it's
                      possible that your timestamps differ.
                      ''')

def feature_loi_loo(self,
                    model_subset=None,
                    param_type=None,
                    param=None,
                    metric=None,
                    baseline=False,
                    baseline_query=None,
                    df=False,
                    figsize=(16,12),
                    fontsize=20):
    '''
    Plot precision across time for the selected model_group_ids, and
    include a leave-one-out, leave-one-in feature analysis to explore the
    leverage/changes of the selected metric across different models.

    This function plots the performance of each model_group_id following an
    user defined performance metric. First, this function check if the
    performance metrics are available to both the models, and the baseline.
    Second, filter the data of interest, and lastly, plot the results as
    timelines (model_id date).

    Arguments:
        -model_subset (list): A list of model_group_ids, in case you want
        to override the selected models.
        - param_type (string): A parameter string with a threshold
        definition: rank_pct, or rank_abs. These are usually defined in the
        postmodeling configuration file.
        - param (int): A threshold value compatible with the param_type.
        This value is also defined in the postmodeling configuration file.
        - metric (string): A string defnining the type of metric use to
        evaluate the model, this can be 'precision@', or 'recall@'.
        - baseline (bool): should we include a baseline for comparison?
        - baseline_query (str): a SQL query that returns the evaluation data from
        the baseline models. This value can also be defined in the
        configuration file.
        - df (bool): If True, no plot is rendered, but a pandas DataFrame
        is returned with the data.
        - figsize (tuple): tuple with figure size parameters
        - fontsize (int): Fontsize for titles

'''

    if model_subset is None:
        model_subset = self.model_group_id

    # Load feature groups and subset
    feature_groups = self.feature_groups
    feature_groups_filter = \
    feature_groups[feature_groups['model_group_id'].isin(model_subset)]

    # Format metrics columns and filter by metric of interest
    model_metrics = self.metrics
    model_metrics[['param', 'param_type']] = \
            model_metrics['parameter'].str.split('_', 1, expand=True)
    model_metrics['param'] =  model_metrics['param'].astype(str).astype(float)
    model_metrics['param_type'] = model_metrics['param_type'].apply(lambda x: 'rank_'+x)

    model_metrics_filter = model_metrics[(model_metrics['metric'] == metric) &
                                         (model_metrics['param'] == param) &
                                         (model_metrics['param_type'] == param_type)].\
            filter(['model_group_id', 'model_id', 'as_of_date_year',
                    'value'])

    # Merge metrics and features and filter by threshold definition
    metrics_merge = model_metrics_filter.merge(feature_groups_filter,
                                              how='inner',
                                              on='model_group_id')

    # LOO and LOI definition
    all_features = set(metrics_merge.loc[metrics_merge['experiment_type'] == 'All features'] \
                       ['feature_group_array'][0])

    metrics_merge_experimental = \
    metrics_merge[metrics_merge['experiment_type'] != 'All features']

    metrics_merge_experimental['feature_experiment'] = \
            metrics_merge_experimental.apply(lambda row: \
                               list(all_features.difference(row['feature_group_array']))[0] + '_loo' \
                               if row['experiment_type'] == 'LOO' \
                               else row['feature_group_array'] + '_loi', axis=1)

    metrics_merge_experimental = \
    metrics_merge_experimental.filter(['feature_experiment',
                                       'as_of_date_year', 'value'])

    if baseline == True:

        baseline_metrics = pd.read_sql(baseline_query, con=self.engine)
        baseline_metrics[['param', 'param_type']] = \
                baseline_metrics['parameter'].str.split('_', 1, expand=True)
        baseline_metrics['param'] = baseline_metrics['param'].astype(str).astype(float)
        baseline_metrics['param_type'] = baseline_metrics['param_type'].apply(lambda x: 'rank_'+x)

        # Filter baseline metrics and create pivot table to join with
        # selected models metrics
        baseline_metrics_filter =  baseline_metrics[(baseline_metrics['metric'] == metric) &
                                                    (baseline_metrics['param'] == param) &
                                                    (baseline_metrics['param_type'] == param_type)].\
            filter(['model_group_id', 'model_id', 'as_of_date_year', 'value'])

        baseline_metrics_filter['feature_experiment'] = \
                baseline_metrics_filter.model_group_id.apply(lambda x: \
                                                             'baseline_' + \
                                                             str(x))

        metrics_merge_experimental = \
                metrics_merge_experimental.append(baseline_metrics_filter,
                                                 sort=True)
    if df == True:
        return metrics_merge_experimental

    else:
        try:
            sns.set_style('whitegrid')
            fig, ax = plt.subplots(figsize=figsize)
            for feature, df in metrics_merge_experimental.groupby(['feature_experiment']):
                ax = df.plot(ax=ax, kind='line',
                                  x='as_of_date_year',
                                  y='value',
                                  label=feature)
            metrics_merge[metrics_merge['experiment_type'] == 'All features']. \
                    groupby(['experiment_type']). \
                    plot(ax=ax,
                         kind='line',
                         x='as_of_date_year',
                         y='value',
                         label='All features')
            plt.title(str(metric).capitalize() +\
                      ' for selected model_groups in time.',
                      fontsize=fontsize)
            ax.tick_params(labelsize=16)
            ax.set_xlabel('Year of prediction (as_of_date)', fontsize=20)
            ax.set_ylabel(f'{str(metric)+str(param_type)+str(param)}',
                          fontsize=20)
            plt.xticks(model_metrics_filter.as_of_date_year.unique())
            plt.yticks(np.arange(0,1,0.1))
            legend=plt.legend(bbox_to_anchor=(1.05, 1),
                       loc=2,
                       borderaxespad=0.,
                       title='Experiment Type',
                       fontsize=fontsize)
            legend.get_title().set_fontsize('16')

        except TypeError:
                print(f'''
                      Oops! model_metrics_pivot table is empty. Several problems
                      can be creating this error:
                      1. Check that {param_type}@{param} exists in the evaluations
                      table
                      2. Check that the metric {metric} is available to the
                      specified {param_type}@{param}.
                      3. You basline model can have different specifications.
                      Check those!
                      4. Check overlap between baseline dates and model dates.
                      The join is using the dates for doing these, and it's
                      possible that your timestamps differ.
                      ''')
