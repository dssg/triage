import json
import yaml
import pandas as pd

from triage.component.model_monitor.mm_utils import get_default_args

pd.options.mode.chained_assignment = None


def non_single_valued_columns(df, dropna=True):
    result_cols = []
    for col in df.columns.values:
        if dropna:
            if len(df[col].dropna().unique()) > 1:
                result_cols.append(col)
        else:
            if len(df[col].unique()) > 1:
                result_cols.append(col)
    return result_cols


def aggregate_feature_groups(feature_list):
    feature_sets = feature_list.apply(lambda l: l.replace('{', '').replace('}', '').split(',')[:-1])
    return set().union(*feature_sets)


def autogenerate_model_group_config(df):
    """
    Generate new config file from existing arrangement
    """

    # clean feature list sets
    # feature_sets = df['feature_list'].apply(lambda l: l.replace('{', '').replace('}', '').split(',')[:-1])
    # global_feature_sets = set().union(*feature_sets)

    # for each model_type, extract all parameter differences
    config_output = dict()
    config_output['model_group_differences'] = dict()

    for model_type, model_groups_by_type in df.groupby('model_type'):
        config_output['model_group_differences'][model_type] = dict()

        # fetch default arguments for base class and apply to model parameters
        default_args = get_default_args(model_type)
        complete_model_parameters = model_groups_by_type['model_parameters'].apply(
            lambda p: dict(default_args, **json.loads(p))
        )

        # identify class-specific differences
        param_df = pd.DataFrame(list(complete_model_parameters))
        config_output['model_group_differences'][model_type]['tracked_parameters'] = non_single_valued_columns(param_df)

        # identify class-specific config differences
        model_config_df = pd.DataFrame(list(model_groups_by_type['model_config']))
        config_output['model_group_differences'][model_type]['tracked_model_config_parameters'] = \
            non_single_valued_columns(model_config_df)

    # find all differences globally in model configuration
    config_output['model_config_params'] = dict()
    model_config_df = pd.DataFrame(list(df['model_config']))
    config_output['model_config'] = non_single_valued_columns(model_config_df)

    return config_output
