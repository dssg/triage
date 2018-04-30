"""
Library for error analysis functions
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd

def reliability_curve(y_true, y_score, bins=10, normalize=False):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities whould be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.

    Parameters
    ----------

    y_true : array[n_samples]
        True binary labels (0 or 1).

    y_score : array[n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=False
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is mapped onto 0 and the largest one
        onto 1.


    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.


    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos


def plot_reliability_curve(y_score,
                           y_score_bin_mean,
                           empirical_prob_pos,
                           no_bins,
                           label='RC curve',
                           savefig='reliability_curve.png'):
    """plots reliability_curve

    Parameters
    ----------
    y_score: array[float]
        score of from the model
    
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.

    Returns
    -------
    reliability_curve: plot

    
    """

    plt.style.use('ggplot')

    y_score_ = y_score
    plt.figure(0, figsize=(8, 8))
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
    scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
    plt.plot(y_score_bin_mean[scores_not_nan],empirical_prob_pos[scores_not_nan], label=label, marker='o')
    plt.ylabel("Empirical probability")
    plt.yticks(np.linspace(0, 0.9, 10) + 0)
    plt.xticks(np.linspace(0, 0.9, 10) + 0)
    plt.legend(loc=0)

    plt.subplot2grid((3, 1), (2, 0))
    #y_score_ = (y_score_ - y_score_.min()) / (y_score_.max() - y_score_.min())
    plt.hist(y_score_, range=(0, 1), bins=no_bins, label=label,
             histtype="step", lw=2)
    plt.xticks(np.linspace(0, 0.9, 10) + 0)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.legend(loc='upper left', ncol=2)

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
    

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
    df_data = pd.DataFrame({'score':y_score,'label':y_label})
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1,figsize=(22, 12))
    sns.set_context("poster", font_scale=2.25, rc={"lines.linewidth": 1.25,"lines.markersize":8})
    plt.hist(df_data[df_data.values==0].score,bins=20,normed=True,alpha=0.5,label='0 class')
    plt.hist(df_data[df_data.values==1].score,bins=20,normed=True,alpha=0.5,label='1 class')
    plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=7,ncol=2, borderaxespad=0.)
    plt.show()
    if savefig:
        logging.info('Saving figure as {}'.format(savefig))
        plt.savefig(savefig)

