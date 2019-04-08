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



def aequitas_audit(df, configs, model_id=1, preprocessed=False):
    """

    :param df:
    :param ref_groups_method:
    :param model_id:
    :param configs:
    :param report:
    :param preprocessed:
    :return:
    """
    if not preprocessed:
        df, attr_cols_input = preprocess_input_df(df)
        if not configs.attr_cols:
            configs.attr_cols = attr_cols_input
    g = Group()
    print('Welcome to Aequitas-Audit')
    print('Fairness measures requested:', ','.join(configs.fair_measures_requested))
    groups_model, attr_cols = g.get_crosstabs(df, score_thresholds=configs.score_thresholds, model_id=model_id,
                                              attr_cols=configs.attr_cols)
    print('audit: df shape from the crosstabs:', groups_model.shape)
    b = Bias()
    # todo move this to the new configs object / the attr_cols now are passed through the configs object...
    ref_groups_method = configs.ref_groups_method
    if ref_groups_method == 'predefined' and configs.ref_groups:
        bias_df = b.get_disparity_predefined_groups(groups_model, df, configs.ref_groups)
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model, df)
    else:
        bias_df = b.get_disparity_min_metric(groups_model, df)
    print('Any NaN?: ', bias_df.isnull().values.any())
    print('bias_df shape:', bias_df.shape)


    f = Fairness(tau=configs.fairness_threshold)
    print('Fairness Threshold:', configs.fairness_threshold)
    print('Fairness Measures:', configs.fair_measures_requested)
    group_value_df = f.get_group_value_fairness(bias_df, fair_measures_requested=configs.fair_measures_requested)
    return group_value_df




def get_models_list(configs, engine):
    models_query = "select model_id from model_metadata.models"
    if configs['model_group_id'] or configs['train_end_time']:
        models_query += " where "
        if configs['model_group_id']:
            models_query += " model_group_id in ({model_group_id_list}) ".format(
                    model_group_id_list = ",".join(configs['model_group_id']))
        if configs['train_end_time']:
            if configs['model_group_id']:
                models_query += " and "
            models_query += " train_end_time in ({train_end_time_list}) ".format(
                    train_end_time_list = ",".join(["'" + str(train_date) + "'::date " for train_date in configs['train_end_time']]))
    models_query += " order by model_id asc"
    models_list = pd.DataFrame.pg_copy_from(models_query, engine)['model_id'].tolist()
    return models_list

def get_predictions(model_id, predictions_schema, config, engine):
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


def manage_audits(df, configs, preprocessed=False):

    group_value_df = None
    if df is not None:
        if 'model_id' in df.columns:
            model_df_list = []
            for model_id in df.model_id.unique():
                model_df = audit(df.loc[df['model_id'] == model_id], model_id=model_id, configs=configs,
                                               preprocessed=preprocessed)
                model_df_list.append(model_df)
            group_value_df = pd.concat(model_df_list)

        else:
            group_value_df, report = audit(df, configs=configs, preprocessed=preprocessed)
    else:
        logging.error('run_csv: could not load a proper dataframe from the input filepath provided.')
    # print(report)
    return group_value_df


def run_bias(engine, config):

    create_tables = 'append'
    if args.create_tables:
        create_tables = 'replace'
    input_query = config.db['input_query']

    df = pd.DataFrame.pg_copy_from(input_query, engine)

    group_value_df = run(df, configs=config, preprocessed=False)
    try:
        group_value_df.set_index(['model_id', 'attribute_name']).pg_copy_to(
            schema=output_schema,
            name=output_table,
            con=engine,
            if_exists=create_tables)
    except SQLAlchemyError as e:
        logging.error('push_db_data: Could not push results to the target database.' + e)


