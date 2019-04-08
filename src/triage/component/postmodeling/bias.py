import logging
import pandas as pd
import ohio.ext.pandas  # noqa

from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df


class AequitasConfigLoader(object):
    original_fairness_measures = (
        'Statistical Parity', 'Impact Parity', 'FDR Parity',
        'FPR Parity', 'FNR Parity', 'FOR Parity', 'TPR Parity',
        'Precision Parity')

    def __init__(self, ref_groups_method='min_metric', fairness_threshold=0.8,
                 attr_cols=None, report=True, score_thresholds=None,
                 ref_groups=None,
                 fairness_measures=original_fairness_measures):
        self.ref_groups_method = ref_groups_method
        self.fairness_threshold = fairness_threshold
        self.attr_cols = attr_cols
        self.report = report
        self.score_thresholds = score_thresholds
        self.ref_groups = ref_groups
        self.fair_measures_requested = list(fairness_measures)



def aequitas_audit(df, configs, model_id, preprocessed=False):
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
    logging.info('Welcome to Aequitas-Audit')
    logging.info('Fairness measures requested:', ','.join(configs.fair_measures_requested))
    groups_model, attr_cols = g.get_crosstabs(df, score_thresholds=configs.score_thresholds, model_id=model_id,
                                              attr_cols=configs.attr_cols)
    logging.info('audit: df shape from the crosstabs:', groups_model.shape)
    b = Bias()
    # todo move this to the new configs object / the attr_cols now are passed through the configs object...
    ref_groups_method = configs.ref_groups_method
    if ref_groups_method == 'predefined' and configs.ref_groups:
        bias_df = b.get_disparity_predefined_groups(groups_model, df, configs.ref_groups)
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model, df)
    else:
        bias_df = b.get_disparity_min_metric(groups_model, df)
    logging.info('Any NaN?: ', bias_df.isnull().values.any())
    logging.info('bias_df shape:', bias_df.shape)


    f = Fairness(tau=configs.fairness_threshold)
    logging.info('Fairness Threshold:', configs.fairness_threshold)
    logging.info('Fairness Measures:', configs.fair_measures_requested)
    group_value_df = f.get_group_value_fairness(bias_df, fair_measures_requested=configs.fair_measures_requested)
    return group_value_df




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


def get_predictions(model_id, predictions_schema, config, engine):
    """

    Args:
        model_id:
        predictions_schema:
        config:
        engine:

    Returns:

    """
    # the entity_id (and as_of_date if wanted) comes from the attributes_query
    predictions_query = """ with attributes as ({attributes_query}) 
                        select p.model_id, p.score, a.*
                        from {predictions_schema}.predictions p
                        left join attributes a using ({join_predictions_on})
                        where model_id = {model_id}
                        order by score desc""".format(
                attributes_query=config['attributes_query'],
                join_predictions_on=",".join(config['join_predictions_on']),
                predictions_schema=predictions_schema,
                model_id=model_id)
    return pd.DataFrame.pg_copy_from(predictions_query, engine)


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
        raise Exception("""
            Postmodeling bias audit: get_models_list() returned empty model ids list.""")
    for model_id in models_list:
        for schema in predictions_schemas:
            logging.info("Running aequitas audit for model_id={model_id} and predictions_schema={schema}".format())
            input_df = get_predictions(model_id, schema, config, engine)
            if not input_df:
                raise Exception("""
                Postmodeling bias audit: get_predictions() returned empty dataframe for model_id = {model_id}. 
                and predictions_schema = {schema}""".format())
            model_audit_df = aequitas_audit(input_df, model_id=model_id, configs=config, preprocessed=False)
            if not model_audit_df:
                raise Exception("""
                Postmodeling bias audit: aequitas_audit() failed. Returned empty dataframe for model_id = {model_id}. 
                and predictions_schema = {schema}""".format())
            model_audit_df.set_index(['model_id', 'attribute_name']).pg_copy_to(
                schema=schema,
                name='bias_audit',
                con=engine,
                if_exists=config['replace_flag'])

