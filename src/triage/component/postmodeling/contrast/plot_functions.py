import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging


def plot_precision_recall_n(model_id, model_name, output_type, engine=None, y_score=None, y_true=None):
    """
    precision and recal at k figure

    Parameters
    ----------
    model_id: int

    model_name: str
        name for the model

    output_type: str
        (save,show) by default save

    y_score, y_true: numpy[array]

    Returns
    -------
    plot.png
    """

    if ((model_id is not None) and (engine is not None)):
        query = 'select score, label_value from results.predictions where model_id = {};'.format(
            model_id)
        df_model = pd.read_sql(query, engine)
        y_score = df_model.score.values
        y_true = df_model.label_value

    from sklearn.metrics import precision_recall_curve

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    sns.set_context("poster", font_scale=1.25, rc={
                    "lines.linewidth": 1.25, "lines.markersize": 8})

    fig, ax1 = plt.subplots(1, figsize=(12, 12))
    sns.set_context("poster", font_scale=1.25, rc={
                    "lines.linewidth": 1.25, "lines.markersize": 8})
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:


plt.show()


def plot_score_dist_classes(model_id, engine):
    """
    score dist plot

    Parameters
    ----------
    model_id: str

    engine: obj

    Returns
    -------
    df_model: DataFrame
        DataFrame for model
    """
    plt.clf()
    query = 'select score, label_value from results.predictions where model_id = {};'.format(
        model_id)
    df_model = pd.read_sql(query, engine)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    sns.set_context("poster", font_scale=1.25, rc={
                    "lines.linewidth": 1.25, "lines.markersize": 8})

    positive = df_model.label_value == 0
    negative = df_model.label_value == 1
    sns.distplot(df_model[positive].score, kde=False)
    sns.distplot(df_model[negative].score, kde=False, color='red')
    plt.xlim(0, 1)
    plt.title(model_id)
    plt.show()
    return df_model


def plot_score_distribution(y_score,
                            y_label,
                            savefig='score_dist.png'):
    """plot score distribution

    Parameters
    ----------
    y_score: array[float]
        score distribution
    y_label: array[int]
        array of labels 1 and 0
    savefig:
        filename to save figure to

    Returns
    -------
    savefig: file
        Saves file to savefig and outputs to screen.
    """
    df_data = pd.DataFrame({'score': y_score, 'label': y_label})
    # sns.set_style("whitegrid")
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, figsize=(22, 12))
    #sns.set_context("poster", font_scale=2.25, rc={"lines.linewidth": 1.25,"lines.markersize":8})
    plt.hist(df_data[df_data.values == 0].score, bins=20,
             normed=True, alpha=0.5, label='0 class')
    plt.hist(list(df_data[df_data.values == 1].score),
             bins=20, normed=True, alpha=0.5, label='1 class')
    plt.legend(bbox_to_anchor=(0., 1.005, 1., .102),
               loc=7, ncol=2, borderaxespad=0.)
    plt.show()
    if savefig:
        logging.info('Saving figure as {}'.format(savefig))
        plt.savefig(savefig)


def plot_induvidial_feature_ranking(train_matrix, test_matrix, entitiy_feature_df, file_name_part='feature_distribution_entitiy_id_'):

    test_matrix_sub_set = pd.merge(
        test_matrix, entitiy_feature_df, left_index=True, right_index=True)

    for index, value in enumerate(entitiy_feature_df.index):
        print('Entity_id: '+str(value))

        # create a figure per entitiy
        fig, axs = plt.subplots(3, 5, figsize=(60, 20))
        axs = axs.ravel()

        for idx, risk_x in enumerate(entitiy_feature_df.columns):

            risk_feature = entitiy_feature_df.loc[value, :][risk_x]

            # get the feature value of this officers in the test-set
            feature_value = test_matrix_sub_set.loc[value, :][risk_feature]

            # train set plot
            axs[idx].hist(train_matrix[train_matrix.outcome == 0][risk_feature],
                          bins=20, normed=True, alpha=0.5, label='0 class')
            axs[idx].hist(train_matrix[train_matrix.outcome == 1][risk_feature],
                          bins=20, normed=True, alpha=0.5, label='1 class')
            axs[idx].axvline(x=feature_value, color='r',
                             linestyle='dashed', linewidth=2)
            axs[idx].set_title('Train: '+str(idx)+' '+str(risk_feature))
            axs[idx].legend()

            # test set plot by test label
            axs[idx+5].hist(test_matrix[test_matrix.outcome == 0][risk_feature],
                            bins=20, alpha=0.5, normed=True, label='0 class')
            axs[idx+5].hist(test_matrix[test_matrix.outcome == 1][risk_feature],
                            bins=20, alpha=0.5, normed=True, label='1 class')
            axs[idx+5].axvline(x=feature_value, color='r',
                               linestyle='dashed', linewidth=2)
            axs[idx+5].set_title('Test: '+str(idx)+' '+str(risk_feature))
            axs[idx+5].legend()

            # test set plot by top_k
            axs[idx+10].hist(test_matrix[risk_feature], bins=20,
                             alpha=0.5, normed=True, label='Everyone')
            axs[idx+10].hist(test_matrix_sub_set[risk_feature], bins=20,
                             alpha=0.5, normed=True, label='Top K entities')
            axs[idx+10].axvline(x=feature_value, color='r',
                                linestyle='dashed', linewidth=2)
            axs[idx+10].set_title('Test topk: '+str(idx)+' '+str(risk_feature))
            axs[idx+10].legend()

            fig.savefig(file_name_part+str(value)+'.png', dpi=100)

        # clear up memory
        plt.close("all")


def plot_feature_distribution_topk(test_matrix, test_matrix_sub_set, topn_feature_list,  savefig='feature_distribution_topk.png'):

    fig, axs = plt.subplots(6, 5, figsize=(60, 60))

    axs = axs.ravel()

    for idx, feature in enumerate(topn_feature_list):
        axs[idx].hist(test_matrix[feature], bins=20,
                      alpha=0.5, normed=True, label='Everyone')
        axs[idx].hist(test_matrix_sub_set[feature], bins=20,
                      alpha=0.5, normed=True, label='TopK')
        axs[idx].set_title(str(idx+1)+' '+str(feature))
        axs[idx].legend()

    #fig.set_size_inches(25, 30)
    if savefig:
        fig.savefig(savefig, dpi=100)
    plt.show()


def plot_feature_distribution(matrix, topn_feature_list,  savefig='feature_distribution.png'):

    fig, axs = plt.subplots(6, 5, figsize=(60, 60))

    axs = axs.ravel()

    for idx, feature in enumerate(topn_feature_list):
        axs[idx].hist(matrix[matrix.outcome == 0][feature],
                      bins=20, normed=True, alpha=0.5, label='0 class')
        axs[idx].hist(matrix[matrix.outcome == 1][feature],
                      bins=20, normed=True, alpha=0.5, label='1 class')
        axs[idx].set_title(str(idx+1)+' '+str(feature))
        axs[idx].legend()

    #fig.set_size_inches(25, 30)
    if savefig:
        fig.savefig(savefig, dpi=100)
    plt.show()
