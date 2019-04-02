# coding: utf-8

def _error_labeler(self,
                  path,
                  param=None,
                  param_type=None):
    '''
    Explore the underlying causes of errors using decision trees to explain the
    residuals base on the same feature space used in the model. This
    exploration will get the most relevant features that determine y - y_hat
    distance and may help to understand the outomes of some models.

    This function will label the errors and return two elements relevant to
    model these. First, a feature matrix (X) with all the features used by
    the model. Second, an iterator with different labeled errors: FPR, FRR,
    and the general error.

    Arguments:
        - param_type: (str) type of parameter to define a threshold. Possible
          values come from triage evaluations: rank_abs, or rank_pct
        - param: (int) value
        - path: path for the ProjectStorage class object
    '''

    test_matrix = self.preds_matrix(path)

    if param_type == 'rank_abs':
        # Calculate residuals/errors
        test_matrix_thresh = test_matrix.sort_values(['rank_abs'], ascending=True)
        test_matrix_thresh['above_thresh'] = \
                np.where(test_matrix_thresh['rank_abs'] <= param, 1, 0)
        test_matrix_thresh['error'] = test_matrix_thresh['label_value'] - \
                test_matrix_thresh['above_thresh']
    elif param_type == 'rank_pct':
        # Calculate residuals/errors
        test_matrix_thresh = test_matrix.sort_values(['rank_pct'], ascending=True)
        test_matrix_thresh['above_thresh'] = \
                np.where(test_matrix_thresh['rank_pct'] <= param, 1, 0)
        test_matrix_thresh['error'] = test_matrix_thresh['label_value'] - \
                test_matrix_thresh['above_thresh']
    else:
        raise AttributeError('''Error! You have to define a parameter type to
                             set up a threshold
                             ''')

    # Define labels using the errors
    dict_errors = {'FP': (test_matrix_thresh['label_value'] == 0) &
                          (test_matrix_thresh['above_thresh'] == 1),
                   'FN': (test_matrix_thresh['label_value'] == 1) &
                          (test_matrix_thresh['above_thresh'] == 0),
                   'TP':  (test_matrix_thresh['label_value'] == 1) &
                          (test_matrix_thresh['above_thresh'] == 1),
                   'TN':  (test_matrix_thresh['label_value'] == 0) &
                          (test_matrix_thresh['above_thresh'] == 0)
                  }
    test_matrix_thresh['class_error'] = np.select(condlist=dict_errors.values(),
                                                 choicelist=dict_errors.keys(),
                                                 default=None)

    # Split data frame to explore FPR/FNR against TP and TN
    test_matrix_thresh_0 = \
    test_matrix_thresh[test_matrix_thresh['label_value'] == 0]
    test_matrix_thresh_1 = \
    test_matrix_thresh[test_matrix_thresh['label_value'] == 1]
    test_matrix_predicted_1 = \
    test_matrix_thresh[test_matrix_thresh['above_thresh'] == 1]

    dict_error_class = {'FPvsAll': (test_matrix_thresh['class_error'] == 'FP'),
                        'FNvsAll': (test_matrix_thresh['class_error'] == 'FN'),
                        'FNvsTP': (test_matrix_thresh_1['class_error'] == 'FN'),
                        'FPvsTN': (test_matrix_thresh_0['class_error'] == 'FP'),
                        'FPvsTP': (test_matrix_predicted_1['class_error'] == 'FP')}

    # Create label iterator
    Y = [(np.where(condition, 1, -1), label) for label, condition in \
         dict_error_class.items()]

    # Define feature space to model: get the list of feature names
    storage = ProjectStorage(path)
    matrix_storage = MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)
    feature_columns = matrix_storage.columns()

    # Build error feature matrix
    matrices = [test_matrix_thresh,
                test_matrix_thresh,
                test_matrix_thresh_1,
                test_matrix_thresh_0,
                test_matrix_predicted_1]
    X = [matrix[feature_columns] for matrix in matrices]

    return zip(Y, X)

def _error_modeler(self,
                  depth=None,
                  view_plots=False,
                  **kwargs):
   '''
   Model labeled errors (residuals) by the error_labeler (FPR, FNR, and
   general residual) using a RandomForestClassifier. This function will
   yield a plot tree for each of the label numpy arrays return by the
   error_labeler (Y).
   Arguments:
       - depth: max number of tree partitions. This is passed directly to
         the classifier.
       - view_plot: the plot is saved to disk by default, but the
         graphviz.Source also allow to load the object and see it in the
         default OS image renderer
       - **kwargs: more arguments passed to the labeler: param indicating
         the threshold value, param_type indicating the type of threshold,
         and the path to the ProjectStorage.
   '''

   # Get matrices from the labeler
   zip_data = self._error_labeler(param_type = kwargs['param_type'],
                              param = kwargs['param'],
                              path=kwargs['path'])

   # Model tree and output tree plot
   for error_label, matrix in zip_data:

       dot_path = 'error_analysis_' + \
                  str(error_label[1]) + '_' + \
                  str(self.model_id) + '_' + \
                  str(kwargs['param_type']) + '@'+ \
                  str(kwargs['param']) +  '.gv'

       clf = tree.DecisionTreeClassifier(max_depth=depth)
       clf_fit = clf.fit(matrix, error_label[0])
       tree_viz = tree.export_graphviz(clf_fit,
                                       out_file=None,
                                       feature_names=matrix.columns.values,
                                       filled=True,
                                       rounded=True,
                                       special_characters=True)
       graph = graphviz.Source(tree_viz)
       graph.render(filename=dot_path,
                    directory='error_analysis',
                    view=view_plots)

       print(dot_path)

def error_analysis(self, threshold, **kwargs):
    '''
    Error analysis function for ThresholdIterator objects. This function
    have the same functionality as the _error.modeler method, but its
    implemented for iterators, which can be a case use of this analysis.
    If no iterator object is passed, the function will take the needed
    arguments to run the _error_modeler.
    Arguments:
        - threshold: a threshold and threshold parameter combination passed
        to the PostmodelingParamters. If multiple parameters are passed,
        the function will iterate through them.
        -**kwags: other arguments passed to _error_modeler
    '''

    error_modeler = partial(self._error_modeler,
                           depth = kwargs['depth'],
                           path = kwargs['path'],
                           view_plots = kwargs['view_plots'])

    if isinstance(threshold, dict):
       for threshold_type, threshold_list in threshold.items():
           for threshold in threshold_list:
               print(threshold_type, threshold)
               error_modeler(param_type = threshold_type,
                             param = threshold)
    else:
       error_modeler(param_type=kwargs['param_type'],
                     param=kwargs['param'])
