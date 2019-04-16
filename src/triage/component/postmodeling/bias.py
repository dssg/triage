import logging
import pandas as pd
import ohio.ext.pandas  # noqa
import yaml
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df


class AequitasConfigLoader(object):
    original_fairness_measures = (
        'Statistical Parity', 'Impact Parity', 'FDR Parity',
        'FPR Parity', 'FNR Parity', 'FOR Parity', 'TPR Parity',
        'Precision Parity')

    def __init__(self, ref_groups_method='min_metric', fairness_threshold=0.8
                 , score_thresholds=None,
                 ref_groups=None,
                 replace_flag=False,
                 join_predictions_on = ['entity_id'],
                 fairness_measures=original_fairness_measures):
        self.ref_groups_method = ref_groups_method
        self.fairness_threshold = fairness_threshold
        self.score_thresholds = score_thresholds
        self.ref_groups = ref_groups
        self.fair_measures_requested = list(fairness_measures)
        self.replace_flag = replace_flag
        self.join_predictions_on = join_predictions_on

def aequitas_audit(df, configs, model_id, schema, replace_flag, engine, subset_hash=None, preprocessed=False):
    """

    Args:
        df:
        configs:
        model_id:
        preprocessed:

    Returns:

    """

    if not preprocessed:
        df, attr_cols_input = preprocess_input_df(df)
        if not configs.attr_cols:
            configs.attr_cols = attr_cols_input
    g = Group()
    groups_model, attr_cols = g.get_crosstabs(df, score_thresholds=configs.score_thresholds, model_id=model_id,
                                              attr_cols=configs.attr_cols)
    b = Bias()
    ref_groups_method = configs.ref_groups_method
    if ref_groups_method == 'predefined' and configs.ref_groups:
        bias_df = b.get_disparity_predefined_groups(groups_model, df, configs.ref_groups)
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model, df)
    else:
        bias_df = b.get_disparity_min_metric(groups_model, df)
    f = Fairness(tau=configs.fairness_threshold)
    group_value_df = f.get_group_value_fairness(bias_df, fair_measures_requested=configs.fair_measures_requested)
    group_value_df['subset_hash'] = subset_hash
    if group_value_df.empty:
        raise ValueError("""
        Postmodeling bias audit: aequitas_audit() failed. Returned empty dataframe for model_id = {model_id}. 
        and predictions_schema = {schema}""".format())
    group_value_df.set_index(['model_id', 'evaluation_start_time', 'evaluation_end_time', 'score_threshold', 'subset_hash', 'attribute_name', 'attribute_value']).pg_copy_to(
        schema=schema,
        name='bias',
        con=engine,
        if_exists=replace_flag)


def get_models_list(config, engine):
    """

    Args:
        config:
        engine:

    Returns:

    """
    models_query = "select model_id from model_metadata.models"
    if config['model_group_id'] or config['train_end_time']:
        models_query += " where "
        if config['model_group_id']:
            models_query += " model_group_id in ({model_group_id_list}) ".format(
                    model_group_id_list = ",".join(config['model_group_id']))
        if config['train_end_time']:
            if config['model_group_id']:
                models_query += " and "
            models_query += " train_end_time in ({train_end_time_list}) ".format(
                    train_end_time_list = ",".join(["'" + str(train_date) + "'::date " for train_date in config['train_end_time']]))
    models_query += " order by model_id asc"
    return pd.DataFrame.pg_copy_from(models_query, engine)['model_id'].tolist()


def get_distinct_evaluations(model_id, schema, engine):
    """

    Args:
        model_id:
        schema:
        engine:

    Returns:

    """
    eval_query = """ select distinct evaluation_start_time, evaluation_end_time
                from {schema}.evaluations
                where model_id = {model_id}""".format(schema=schema, model_id=model_id)
    evals_df = pd.DataFrame.pg_copy_from(eval_query, engine)
    if evals_df.empty:
        raise ValueError("No evalautions were found for model_id = {model_id} in the schema {schema}.".format())
    return evals_df


def get_distinct_subsets(model_id, evaluation_start_time, evaluation_end_time, schema, engine):
    """

    Args:
        model_id:
        schema:
        engine:

    Returns:

    """
    eval_query = """ select distinct subset_hash
                from {schema}.evaluations
                where model_id = {model_id} and evaluation_start_time='{evaluation_start_time}'::date 
                 and evaluation_end_time = '{evaluation_end_time}'::date 
                 and subset_hash is not null""".format(schema=schema,
                                                    model_id=model_id,
                                                    evaluation_start_time=evaluation_start_time,
                                                    evaluation_end_time=evaluation_end_time)
    return list(pd.DataFrame.pg_copy_from(eval_query, engine)['subset_hash'])


def get_input_df(model_id, schema, evaluation_start_time, evaluation_end_time, config, engine):
    """

    Args:
        model_id:
        schema:
        config:
        engine:

    Returns:

    """
    # the entity_id (and as_of_date if wanted) comes from the attributes_query
    input_query = """ with attributes as ({attributes_query}) 
                        select p.model_id, 
                        '{evaluation_start_time}'::date as evaluation_start_time,
                        '{evaluation_end_time}'::date as evaluation_end_time,
                        p.score, 
                        coalesce(rank_abs, rank() over(order by score desc)) as rank_abs,
                        coalesce(rank_pct, percent_rank() over(order by score desc)) * 100 as rank_pct,
                        p.label_value,
                        a.*
                        from {predictions_schema}.predictions p
                        left join attributes a using ({join_predictions_on})
                        where p.model_id = {model_id}
                        and as_of_date <@ ('{evaluation_start_time}'::date, ('{evaluation_end_time}'::date + interval '1 days')::date)
                        and p.label_value is not null
                        order by p.score desc""".format(
                attributes_query=config['attributes_query'],
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                join_predictions_on=",".join(config['join_predictions_on']),
                predictions_schema=schema,
                model_id=model_id)

    input_df = pd.DataFrame.pg_copy_from(input_query, engine)
    if input_df.empty:
        raise ValueError("No input_df for model_id = {model_id} and schema {schema}.".format())
    return input_df


def get_subset_df(subset_hash, engine):
    metadata_query = """select config from model_metadata.subsets
                            where subset_hash='{subset_hash}'""".format(subset_hash=subset_hash)
    subset_config = yaml.load(pd.DataFrame.pg_copy_from(metadata_query, engine)['config'])
    subset_query = """select entity_id, as_of_date
                        from public.{subset_table} where active = true
                        """.format(subset_table = "subset_{name}_{subset_hash}".format(
                                name=subset_config['name'], subset_hash=subset_config['subset_hash']))
    subset_df = pd.DataFrame.pg_copy_from(subset_query, engine)
    if subset_df.empty:
        raise ValueError("Subset with hash {subset_hash) return no entities from subset table.".format(subset_hash=subset_hash))
    return subset_df



def run_bias(engine, config, predictions_schemas=['test_results', 'train_results']):
    """

    Args:
        engine:
        config:
        predictions_schemas:

    Returns:

    """
    models_list = get_models_list(config)
    if not models_list:
        raise ValueError("""
            Postmodeling bias audit: get_models_list() returned empty model ids list.""")
    for model_id in models_list:
        for schema in predictions_schemas:
            evals_df = get_distinct_evaluations(model_id, schema, engine)
            logging.info("Running aequitas audit for model_id={model_id} and predictions_schema={schema}".format())
            for ind, row in evals_df.iterrows():
                try:
                    input_df = get_input_df(model_id, schema, row['evaluation_start_time'], row['evaluation_end_time'], config, engine)
                    aequitas_audit(input_df, model_id=model_id, schema=schema,
                                                    replace_flag=config['replace_flag'], preprocessed=False)
                    subsets_list = get_distinct_subsets(model_id, row['evaluation_start_time'], row['evaluation_end_time'], config, engine)
                    if subsets_list:
                        for subset_hash in subsets_list:
                            try:
                                subset_df = get_subset_df(subset_hash, engine)
                                input_subset_df = input_df.join(subset_df,how='inner', on=['entity_id','as_of_date'])
                                aequitas_audit(input_subset_df, model_id=model_id, schema=schema,
                                               replace_flag=config['replace_flag'],subset_hash=subset_hash, preprocessed=False)
                            except ValueError:
                                continue
                except ValueError:
                    continue


