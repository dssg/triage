# coding: utf-8

def plot_jaccard_preds(self,
                       param_type=None,
                       param=None,
                       model_subset=None,
                       temporal_comparison=False,
                       figsize=(24, 10),
                       fontsize=12):

    if model_subset is None:
        model_subset = self.model_id

    preds = self.predictions
    preds_filter = preds[preds['model_id'].isin(self.model_id)]

    if temporal_comparison == True:
        try:
            fig = plt.figure(figsize=figsize)
            for key, values in \
            self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
                preds_filter_group = \
                preds_filter[preds_filter['model_id'].isin(values[1])]
                # Filter predictions dataframe by individual dates
                if param_type == 'rank_abs':
                    df_preds_date = preds_filter_group.copy()
                    df_preds_date['above_tresh'] = \
                            np.where(df_preds_date['rank_abs'] <= param, 1, 0)
                    df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                     columns='model_id',
                                                     values='above_tresh')
                elif param_type == 'rank_pct':
                    df_preds_date = preds_filter_group.copy()
                    df_preds_date['above_tresh'] = np.where(df_preds_date['rank_pct'] <= param, 1, 0)
                    df_preds_date['new_entity_id'] = df_preds_date['entity_id'].astype(str) + ":" + df_preds_date['as_of_date'].astype(str)
                    df_sim_piv = df_preds_date.pivot(index='new_entity_id',
                                                     columns='model_id',
                                                     values='above_tresh')
                else:
                    raise AttributeError('''Error! You have to define a parameter type to
                                         set up a threshold
                                         ''')
                        # Calculate Jaccard Similarity for the selected models
                res = pdist(df_sim_piv.T, 'jaccard')
                df_jac = pd.DataFrame(1-squareform(res),
                                  index=preds_filter_group.model_id.unique(),
                                  columns=preds_filter_group.model_id.unique())
                mask = np.zeros_like(df_jac)
                mask[np.triu_indices_from(mask, k=1)] = True

                # Plot matrix heatmap
                ax = fig.add_subplot(np.ceil(self.same_time_models.shape[0]/4), 4, key+1)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title(f'''(as_of_date:{values[0]})''', fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask,
                            cmap='Greens',
                            vmin=0,
                            vmax=1,
                            annot=True,
                            linewidth=0.1)
        except ValueError:
            print(f'''
                  Temporal comparison can be only made for more than one
                  model group.
                 ''')

    else:
            # Call predicitons
            if param_type == 'rank_abs':
                df_preds_date = preds_filter.copy()
                df_preds_date['above_tresh'] = \
                        np.where(df_preds_date['rank_abs'] <= param, 1, 0)
                df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                 columns='model_id',
                                                 values='above_tresh')
            elif param_type == 'rank_pct':
                df_preds_date = preds_filter.copy()
                df_preds_date['above_tresh'] = \
                        np.where(df_preds_date['rank_pct'] <= param, 1, 0)
                df_preds_date['new_entity_id'] = df_preds_date['entity_id'].astype(str) + ":" + df_preds_date['as_of_date'].astype(str)
                df_sim_piv = df_preds_date.pivot(index='new_entity_id',
                                                 columns='model_id',
                                                 values='above_tresh')
            else:
                raise AttributeError('''Error! You have to define a parameter type to
                                     set up a threshold
                                     ''')

            # Calculate Jaccard Similarity for the selected models
            res = pdist(df_sim_piv[model_subset].T, 'jaccard')
            df_jac = pd.DataFrame(1-squareform(res),
                                  index=model_subset,
                                  columns=model_subset)
            mask = np.zeros_like(df_jac)
            mask[np.triu_indices_from(mask, k=1)] = True

            # Plot matrix heatmap
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Model Id', fontsize=fontsize)
            ax.set_ylabel('Model Id', fontsize=fontsize)
            plt.title('Jaccard Similarity Matrix Plot', fontsize=fontsize)
            sns.heatmap(df_jac,
                        mask=mask,
                        cmap='Greens',
                        vmin=0,
                        vmax=1,
                        annot=True,
                        linewidth=0.1)


def plot_jaccard_features(self,
                          top_n_features=10,
                          model_subset=None,
                          temporal_comparison=False,
                          figsize=(30, 10),
                          fontsize=12):

    if model_subset is None:
        model_subset = self.model_id

    f_importances = self.feature_importances
    f_importances_filter = \
    f_importances[f_importances['model_id'].isin(model_subset)]

    if temporal_comparison == True:
        try:
            fig = plt.figure(figsize=figsize)
            for key, values in \
            self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
                f_imp_filter_group = \
                f_importances_filter[f_importances_filter['model_id'].isin(values[1])]

                if top_n_features is not None:
                    f_imp_date = f_imp_filter_group.copy()
                    f_imp_date_filter = \
                            f_imp_filter_group.sort_values('rank_abs')
                    f_imp_date_filter_top = \
                            f_imp_date_filter[f_imp_date_filter['rank_abs']
                                             <= top_n_features]

                    df_sim_piv = f_imp_date_filter_top.pivot(index='feature',
                                                             columns='model_id',
                                                             values='rank_abs')
                else:
                    raise AttributeError('''Error! You have to define a top_n features to
                                         set up a threshold
                                         ''')

                # Calculate Jaccard Similarity for the selected models
                res = pdist(df_sim_piv.T, 'jaccard')
                df_jac = pd.DataFrame(1-squareform(res),
                                  index=values[1],
                                  columns=values[1])
                mask = np.zeros_like(df_jac)
                mask[np.triu_indices_from(mask, k=1)] = True

                # Plot matrix heatmap
                ax = fig.add_subplot(np.ceil(self.same_time_models.shape[0]/4), 4, key+1)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title(f'''(as_of_date:{values[0]})''', fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask,
                            cmap='Greens',
                            vmin=0,
                            vmax=1,
                            annot=True,
                            linewidth=0.1)
        except ValueError:
            print(f'''
                  Temporal comparison can be only made for more than one
                  model group.
                 ''')

    else:
            # Call predicitons
            if top_n_features is not None:
                f_importances_filter_all = f_importances_filter.copy()
                f_importance_filter_all_rank = \
                        f_importances_filter_all.sort_values('rank_abs')
                f_importance_filter_all_rank_top = \
                        f_importance_filter_all_rank[f_importance_filter_all_rank['rank_abs']
                                                    <= top_n_features]

                df_sim_piv = \
                f_importance_filter_all_rank_top.pivot(index='feature',
                                                       columns='model_id',
                                                       values='rank_abs')
            else:
                raise AttributeError('''Error! You have to define a parameter type to
                                     set up a threshold
                                     ''')

            # Calculate Jaccard Similarity for the selected models
            res = pdist(df_sim_piv[model_subset].T, 'jaccard')
            df_jac = pd.DataFrame(1-squareform(res),
                                  index=model_subset,
                                  columns=model_subset)
            mask = np.zeros_like(df_jac)
            mask[np.triu_indices_from(mask, k=1)] = True

            # Plot matrix heatmap
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Model Id', fontsize=fontsize)
            ax.set_ylabel('Model Id', fontsize=fontsize)
            plt.title('Jaccard Similarity Matrix Plot', fontsize=fontsize)
            sns.heatmap(df_jac,
                        mask=mask,
                        cmap='Greens',
                        vmin=0,
                        vmax=1,
                        annot=True,
                        linewidth=0.1)


def _plot_preds_compare_score_dist(self,
                                   m0,
                                   m1,
                                   df_preds_date,
                                   colors=['blue', 'orange'],
                                   bins=np.arange(0,1.01,0.01)):

    '''
    Plotting function for comparing prediction distributions across models.
    This function takes two model_ids with predictions in the same
    prediction window and shows the relative distribution of
    the first model top-k in the second model score distribution.

    This function is meant to be used as a helper function of
    plot_preds_comparisons.

    Arguments:
        - m0, m1: (int) model_id
        - df_preds_date: (dataframe) predictions dataframe
        - colors: (str) color strings. Defaults are blue and orange
        - bins: (np array) number of bins to pass to the seaborn histogram
        plotting function

    Returns:
        matplotlib plot object
    '''

    df_preds_m0 = df_preds_date[df_preds_date['model_id']==m0]
    df_preds_m1 = df_preds_date[df_preds_date['model_id']==m1]

    sns.distplot(df_preds_m0[df_preds_m0['above_tresh']==0]['score'],
                 kde=False,
                 bins=bins,
                 color='grey',
                 label="model " + str(m0) + " predicted label = 0")
    sns.distplot(df_preds_m0[df_preds_m0['above_tresh']==1]['score'],
                 kde=False,
                 bins=bins,
                 color=colors[1],
                 label="model " + str(m0) + " predicted label = 1")

    df_alt_model_scores = \
            pd.merge(df_preds_m0, df_preds_m1[df_preds_m1.above_tresh==1][['entity_id', 'as_of_date']])

    sns.distplot(df_alt_model_scores['score'],
                 kde=False,
                 bins=bins,
                 color=colors[0],
                 label="model " + str(m1) + " predicted label = 1")

    plt.xlabel("Scores from model " + str(m0))
    plt.legend()

def _plot_preds_compare_rank(self,
                             m0,
                             m1,
                             df_preds_date,
                             colors=['black'],
                             show_tp_fp=False,
                             bins = np.arange(0,110,10)):
    '''
    Plot predictions rank comparison for two selected models.

    This function will rank the predictions from one model into the decile
    distribution of the second one. This function is meant to be used as a
    part of the plot_preds_comparison function.

    Arguments:
        - m0, m1: (int) model_ids to compare, only two.
        - df_preds_date: (dataframe) predictions dataframe
        - colors: (str) color string. Default is black
        - show_tp_tn: (bool) Plot true positive and true negatives in the
                       rank distribution plot. Default is False
        - bins: (np array) Number of bins to pass to the seaborn
        histogram plot function.
    '''

    df_preds_m0 = df_preds_date[df_preds_date['model_id']==m0]
    df_preds_m1 = df_preds_date[df_preds_date['model_id']==m1]
    df_alt_model_rank = \
            pd.merge(df_preds_m0, df_preds_m1[df_preds_m1.above_tresh==1][['entity_id', 'as_of_date']])

    if show_tp_fp:
        sns.distplot(df_alt_model_rank[df_alt_model_rank['label_value']==0]['rank_pct'],
                     kde=False,
                     bins=bins,
                     hist=True,
                     color=colors[0],
                     label="false positives")
        sns.distplot(df_alt_model_rank[df_alt_model_rank['label_value']==1]['rank_pct'],
                     kde=False,
                     bins=bins,
                     hist=True,
                     color=colors[1],
                     label="true positives")
        plt.legend()
    else:
        sns.distplot(df_alt_model_rank['rank_pct'],
                     kde=False,
                     bins=bins,
                     hist=True,
                     color=colors[0])
    plt.xlabel("Percentile Rank in model " + str(m0))
    plt.title("model "+str(m1)+" predicted label = 1")
    plt.xticks(bins)

def plot_preds_comparison(self,
                          param_type=None,
                          param=None,
                          model_subset=None,
                          figsize=(28, 16),
                          fontsize=12):
    '''
    Plot predictor distribution comparison (distribution and rank)

    This function compares the predictions of all models, or a subset passed to
    model_subset. To compare predictions, the function will show the
    relative position of the score distribution of the top-k of one of the
    models into another model.

    Also, to compare how "off" can predictions be, the function will plot
    the rank position of the predictions of one model in to another. The
    plot will show the decile position of one model into the other.

    Arguments:
        - param_type: (str) parameter type (i.e. 'rank_abs', 'rank_pct')
        - param: (int) parameter threshold
        - model_subset: (list) list of model_ids to compare. Default is
        none, and the function will take all the models in self.model_id
        - figsize, fontsize: aesthetics for plots.
    '''

    if model_subset is None:
        model_subset = self.model_id

    preds = self.predictions
    preds_filter = preds[preds['model_id'].isin(self.model_id)]

    fig = plt.figure(figsize=figsize)
    try:
        for key, values in \
            self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
            preds_filter_group = \
                                 preds_filter[preds_filter['model_id'].isin(values[1])]
            # Filter predictions dataframe by individual dates
            df_preds_date = preds_filter_group.copy()
            if param_type == 'rank_abs':
                df_preds_date['above_tresh'] = \
                    np.where(df_preds_date['rank_abs'] <= param, 1, 0)
            elif param_type == 'rank_pct':
                df_preds_date['above_tresh'] = \
                    np.where(df_preds_date['rank_pct'] <= param, 1, 0)

            else:
                raise AttributeError('''Error! You have to define a parameter type to
                set up a threshold
                ''')

            sns.set_style('whitegrid')
            sns.set_context("poster", font_scale=1.25, rc={"lines.linewidth": 2.25,"lines.markersize":12})
            plt.clf()
            fig = plt.figure(figsize=figsize)
            for pair in itertools.combinations(values['model_id_array'], 2):
                m0 = pair[0]
                m1 = pair[1]
                colors = {m0: 'blue', m1: 'orange'}
                ax1 = plt.subplot(231)
                self._plot_preds_compare_score_dist(pair[0], pair[1],
                                                   df_preds_date,
                                                   colors=[colors[m0], colors[m1]])
                ax1 = plt.subplot(234)
                self._plot_preds_compare_score_dist(pair[1],
                                                   pair[0],
                                                   df_preds_date,
                                                   colors=[colors[m1], colors[m0]])
                ax1 = plt.subplot(232)
                self._plot_preds_compare_rank(pair[0],
                                             pair[1],
                                             df_preds_date,
                                             colors=[colors[m0]])
                ax1 = plt.subplot(235)
                self._plot_preds_compare_rank(pair[1],
                                             pair[0],
                                             df_preds_date,
                                             colors=[colors[m1]])
                ax1 = plt.subplot(233)
                self._plot_preds_compare_rank(pair[0],
                                             pair[1],
                                             df_preds_date,
                                             show_tp_fp=True,
                                             colors=['lightblue', 'darkblue'])
                ax1 = plt.subplot(236)
                self._plot_preds_compare_rank(pair[1],
                                             pair[0],
                                             df_preds_date,
                                             show_tp_fp=True,
                                             colors=['khaki', 'darkorange'])
            plt.tight_layout()
            fig.suptitle(values['train_end_time'])
            plt.show()
    except ValueError:
        print(f'''
        Temporal comparison can be only made for more than one
        model group.
        ''')
