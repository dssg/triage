import pandas as pd
import numpy as np

def individual_feature_ranking(rftree_feature_list, test_matrix):
        ###################
        # This method is a beta version tested for top k optimized random forests
        ###################

        # create a new text matrix with float conversion (panda stores some columns from postgres as decimal object)
        tmp_test = test_matrix.applymap(lambda x: float(x))
        test_matrix_reduced = pd.DataFrame()

        # add the top n_ranks features from the RandomForest used without dummies
        for feature in rftree_feature_list:
            test_matrix_reduced[feature] = tmp_test[feature]


        test_matrix_rank_distance = pd.DataFrame()

        for feature in test_matrix_reduced.columns:
            feature_median = test_matrix_reduced[feature].median()
            test_matrix_reduced[feature + '_rank'] = test_matrix_reduced[feature].rank()
            # the index of the median value
            idx = np.nanargmin(np.abs(test_matrix_reduced[feature] - feature_median))
            median_rank = test_matrix_reduced[feature + '_rank'].iloc[idx]
            test_matrix_rank_distance[feature] = np.abs(test_matrix_reduced[feature + '_rank'] - median_rank)

        test_matrix_rank_distance = test_matrix_rank_distance.reset_index(drop=True)

        # find the top 5 features where the distance to the median rank value (for standardisation) is the highest
        # across the top n features
        matrix_transposed = test_matrix_rank_distance.T
        result = pd.DataFrame(np.zeros((0, 5)), columns=['risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5'])
        for i in matrix_transposed.columns:
            df1row = pd.DataFrame(matrix_transposed.nlargest(5, i).index.tolist(),
                                  index=['risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5']).T
            result = pd.concat([result, df1row], axis=0)
        result = result.reset_index(drop=True)

        # prepare dataframe return
        result['entity_id'] = test_matrix_reduced.index.values
        result=result.set_index('entity_id')

     
        return result
