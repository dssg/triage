'''
Auxiliary functions and helpers:

This set of functions are helper functions to format data 
(i.e., prediction matrices, etc.) for plotting. This functions
are called in both Model class and ModelGroup class in 
evaluation.py.
'''

import pandas as pd


# modifies df_dest directly
# assumes categorical var columns are mutually exclusive
def recombine_categorical(df_dest, df_source, prefix, suffix='', entity_col='entity_id'):
    """Combine a categorical variable that has been one-hot encoded into binary columns
       back into a single categorical variable (assumes the naming convention of collate).

       The binary columns here are assumed to be mutually-exclusive and each entity will
       only be given (at most) a single categorical value (in practice, this will be the
       last such value encountered by the for loop).

       Note: modifies the input data frame, df_dest, directly.

    Arguments:
        df_dest (DataFrame) -- data frame into which the recombined categorical variable will be stored
        df_source (DataFrame) -- data frame with the one-hot encoded source columns
        prefix (string) -- prefix shared by the binary columns for the categorical variable, typically
                           something like 'feature_group_entity_id_1y_columnname_'
        suffix (string) -- suffix shared by the binary columns for the categorical variable, typically
                           something like '_min', '_max', '_avg', etc.
        entity_col (string) -- column to identify entities to map the 0/1 binary values in the one-hot
                               columns to catagorical values in the recombined column.

    Returns: tuple of the modified df_dest and the new categorical column name
    """
    cat_col = prefix+suffix+'_CATEGORIES'
    df_dest[cat_col] = np.NaN
    df_dest[cat_col] = df_dest[cat_col].astype('category')

    for col in df_source.columns[df_source.columns.str.startswith(prefix) & df_source.columns.str.endswith(suffix)]:
        cat_val = col.replace(prefix, '').replace(suffix, '')
        df_dest[cat_col].cat.add_categories([cat_val], inplace=True)
        cat_entities = df_source.loc[df_source[col]==1][entity_col]
        df_dest.loc[df_dest[entity_col].isin(cat_entities), cat_col] = cat_val

        return (df_dest, cat_col)

def pd_int_var(df, integer, var_name):
    df[var_name] = integer
    return df

