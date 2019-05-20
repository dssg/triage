import pandas
import sqlalchemy

from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df

from triage.component.catwalk.storage import MatrixStore
from triage.util.db import scoped_session


def query_protected_groups_table(
    db_engine,
    as_of_dates,
    protected_group_table_name,
    labels,
    cohort_hash,
):
    """Queries the protected groups table to retrieve the protected attributes for each date
    Args:
        db_engine (sqlalchemy.engine) a database engine
        as_of_dates (list) the as_of_Dates to query
        protected_group_table_name (str) the name of the table to query

    Returns: (pandas.DataFrame) a dataframe indexed by the entity-date pairs
        active in the subset
    """
    as_of_dates_sql = "[{}]".format(
        ", ".join("'{}'".format(date.strftime("%Y-%m-%d %H:%M:%S.%f")) for date in as_of_dates)
    )
    query_string = f"""
        with dates as (
            select unnest(array{as_of_dates_sql}::timestamp[]) as as_of_date
        )
        select *
        from {protected_group_table_name}
        join dates using(as_of_date)
        where cohort_hash = '{cohort_hash}'
    """
    protected_df = pandas.DataFrame.pg_copy_from(
        query_string,
        engine=db_engine,
        parse_dates=["as_of_date"],
        index_col=MatrixStore.indices,
    )
    del protected_df['cohort_hash']
    if protected_df.empty:
        return None
    else:
        return protected_df.align(labels, join="inner", axis=0)[0]


def bias_audit(
    db_engine,
    model_id,
    protected_df,
    predictions_proba,
    labels,
    tie_breaker,
    subset_hash,
    bias_config,
    matrix_type,
    evaluation_start_time,
    evaluation_end_time,
    matrix_uuid
):
    """
    Runs the bias audit and saves the result in the bias table.

    Args:
        model_id:
        protected_df:
        predictions_proba:
        labels:
        tie_breaker:
        subset_hash:
        bias_config:
        matrix_type:
        evaluation_start_time:
        evaluation_end_time:
        matrix_uuid:

    Returns:

    """
    if protected_df.empty:
        return

    protected_df = protected_df.copy()
    protected_df['model_id'] = model_id
    protected_df['score'] = predictions_proba
    protected_df['label_value'] = labels
    # to preprocess aequitas requires the following columns:
    # score, label value, model_id, protected attributes 
    df, attr_cols_input = preprocess_input_df(protected_df)
    g = Group()
    score_thresholds = {}
    score_thresholds['rank_abs'] = bias_config['thresholds'].get('top_n', [])
    score_thresholds['rank_pct'] = bias_config['thresholds'].get('percentiles', [])
    groups_model, attr_cols = g.get_crosstabs(df,
                                              score_thresholds=score_thresholds,
                                              model_id=model_id,
                                              attr_cols=attr_cols_input)
    b = Bias()
    ref_groups_method = bias_config.get('ref_groups_method', None)
    if ref_groups_method == 'predefined' and bias_config['ref_groups']:
        bias_df = b.get_disparity_predefined_groups(groups_model, df, bias_config['ref_groups'])
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model, df)
    else:
        bias_df = b.get_disparity_min_metric(groups_model, df)
    f = Fairness(tau=0.8) # the default fairness threshold is 0.8
    group_value_df = f.get_group_value_fairness(bias_df)
    group_value_df['subset_hash'] = subset_hash
    group_value_df['tie_breaker'] = tie_breaker
    group_value_df['evaluation_start_time'] = evaluation_start_time
    group_value_df['evaluation_end_time'] = evaluation_end_time
    group_value_df['matrix_uuid'] = matrix_uuid
    group_value_df = group_value_df.rename(index=str, columns={"score_threshold": "parameter"})
    #group_value_df['parameter'] =
    #delete score_thresholds
    if group_value_df.empty:
        raise ValueError("""
        Bias audit: aequitas_audit() failed.
        Returned empty dataframe for model_id = {model_id}, and subset_hash = {subset_hash}
        and predictions_schema = {schema}""".format())
    with scoped_session(db_engine) as session:
        for index, row in group_value_df.iterrows():
            session.query(matrix_type.aequitas_obj).filter_by(
                model_id=model_id,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                subset_hash=subset_hash,
                parameter=row['parameter'],
                tie_breaker=tie_breaker,
                matrix_uuid=matrix_uuid,
                attribute_name=row['attribute_name'],
                attribute_value=row['attribute_value']
            ).delete()
        session.bulk_insert_mappings(matrix_type.aequitas_obj, group_value_df.to_dict(orient="records"))
