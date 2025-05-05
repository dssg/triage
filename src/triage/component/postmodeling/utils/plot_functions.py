import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from triage.component.catwalk.utils import sort_predictions_and_labels


def plot_score_distribution(scores, nbins=10, ax=None, title=None):
    """ Plot the distribution of predicted scores.

    Args:
        scores (pd.DataFrame): Dataframe with predicted scores and ranks -- output of ModelAnalyzer.get_predictions()
        nbins (int, optional): Number of bins for the histogram. Defaults to 10.
        topk (int, optional): If provided, only the top k scores will be plotted.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.

    Returns:
        None
    """
      
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
            
    ax.hist(scores.score,
                bins=nbins,
                alpha=0.5,
                color='blue')
    
    ax.axvline(scores.score.mean(),
                color='black',
                linestyle='dashed')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Score')
    
    if title is not None:
        ax.set_title(title)
    
    
def plot_score_distribution_by_label(scores, nbins, ax=None, title=None):
    """ Plot the distribution of predicted scores by label.

    Args:
        scores (pd.DataFrame): Dataframe with predicted scores and ranks -- output of ModelAnalyzer.get_predictions()
        nbins (int, optional): Number of bins for the histogram. Defaults to 10.
        topk (int, optional): If provided, only the top k scores will be plotted.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.

    Returns:
        None
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    grp_obj = scores.groupby('label_value')
    
    colors = sns.color_palette("tab10", len(grp_obj))

    for i, (label, group) in enumerate(grp_obj):
        ax.hist(
            group.score, 
            bins=nbins,
            alpha=0.5,
            color=colors[i],
            label=label
        )
        
        ax.axvline(
            group.score.mean(),
            color=colors[i],
            linestyle='dashed'
        )

    ax.legend()
    
# TODO: Facilitate plotting pr-k with absolute thresholds
def plot_precision_recall_at_k(predictions=None, evaluations=None, k_upper_bound=1, step_size=0.01, only_recall=False, ax=None):
    """ Plot precision and recall at k. This function can be used to plot PR-k for subsets as well (using the proper evaluations dataframe)

    Args:
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.
        k_upper_bound (int, optional): Upper bound for k. Defaults to 1.
        step_size (float, optional): Step size for k. Defaults to 0.01.
        subset_hash (str, optional): Hash of the subset to plot. Defaults to None. Need to provide evaluations and subset_hash        only_recall (bool, optional): If True, only plot recall. Defaults to False.

    Returns:
        None
    """
    
    if predictions is None and evaluations is None:
        raise ValueError("Either the predictions or evaluations must be provided.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
        
    # Removing entities with null labels if any
    if predictions is not None: 
        metrics = list()
        predictions.dropna(axis=0, inplace=True, subset=['label_value'])
        
        ks = np.arange(0, k_upper_bound + step_size, step_size)
        
        for k in ks:
            topk = predictions[predictions.rank_pct_no_ties <= k]
            
            d = dict()
            d['k'] = k
            d['precision'] = topk.label_value.mean()
            d['recall'] = topk.label_value.sum() / predictions.label_value.sum()
            metrics.append(d)
            
        metrics = pd.DataFrame(metrics)
        
        if not only_recall:
            sns.lineplot(data=metrics, x='k', y='precision', ax=ax, label='Precision')
        
        sns.lineplot(data=metrics, x='k', y='recall', ax=ax, label='Recall')
        ax.set_xlabel('% of Cohort (k)')
        ax.set_ylabel('Value')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
    if evaluations is not None:
        
        evaluations['perc_points'] = [x.split('_')[0] for x in evaluations['parameter'].tolist()]
        evaluations['perc_points'] = pd.to_numeric(evaluations['perc_points'])

        msk_prec = evaluations['metric']=='precision@'
        msk_recall = evaluations['metric']=='recall@'
        msk_pct = evaluations['parameter'].str.contains('pct')
    
        # plot precision
        if not only_recall:
            sns.lineplot(
                x='perc_points',
                y='metric_value', 
                data=evaluations[msk_pct & msk_prec], 
                label='precision@k',
                ax=ax, 
                estimator='mean', ci='sd'
            )
        # plot recall
        sns.lineplot(
            x='perc_points', 
            y='metric_value', 
            data=evaluations[msk_pct & msk_recall], 
            label='recall@k', 
            ax=ax, 
            estimator='mean', 
            ci='sd'
        )
        
        ax.set_xlabel('% of Cohort (k)')
        ax.set_ylabel('Value')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
    return ax
            
        
def plot_feature_importance(feature_importance, ax=None, n_top_features=20, **kwargs):
    """ Plot feature importance.

    Args:
        feature_importance (pd.DataFrame): Dataframe with feature importance -- output of ModelAnalyzer.get_feature_importance()
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.
        n_top_features (int, optional): Number of top features to plot. Defaults to 20.

    Returns:
        None
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(n_top_features)
    feature_importance['feature_display'] = feature_importance['feature'].str.replace('_entity_id', '')
    
    sns.barplot(
        data=feature_importance,
        x='importance',
        y='feature',
        ax=ax,
        color='red',
        **kwargs
    )
    
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')    
    
    return ax
    