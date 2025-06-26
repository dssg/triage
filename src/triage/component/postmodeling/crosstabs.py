import ohio.ext.pandas
import pandas as pd
import verboselogs, logging

logger = verboselogs.VerboseLogger(__name__)

from scipy import stats
import yaml
import psycopg2
from io import StringIO

from triage.component.architect.database_reflection import table_exists
from triage.component.catwalk.storage import ProjectStorage
from triage.component.catwalk.utils import save_db_objects


class CrosstabsConfigLoader:
    def __init__(self, config_path=None, config=None):
        self.output = {}
        self.models_list_query = None
        self.as_of_dates_query = None
        self.models_dates_join_query = None
        self.features_query = None
        self.predictions_query = None

        if config:
            config_to_load = config
        elif config_path:
            with open(config_path, "r") as stream:
                config_to_load = yaml.full_load(stream)
        else:
            raise ValueError("Either config_path or config are needed")
        self.__dict__.update(config_to_load)


# Later to be run against feature matrices of high- and low-risk entities.
# Each function receives two arguments: A dataframe of high-risk entities'
# features (as columns), and a dataframe of low-risk entities' features.
# The functions return either a Series, where the row index are the names of
# the feature columns, or a dataframe, again with the rows indexed by feature
# names.

high_risk_n = lambda hr, lr: pd.Series(index=hr.columns, data=hr.shape[0])
low_risk_n = lambda hr, lr: pd.Series(index=lr.columns, data=lr.shape[0])
high_risk_mean = lambda hr, lr: hr.mean(axis=0)
low_risk_mean = lambda hr, lr: lr.mean(axis=0)
high_risk_std = lambda hr, lr: hr.std(axis=0)
low_risk_std = lambda hr, lr: lr.std(axis=0)
hr_lr_ratio = lambda hr, lr: hr.mean(axis=0) / lr.mean(axis=0)
high_risk_support = lambda hr, lr: (hr > 0).sum(axis=0)
low_risk_support = lambda hr, lr: (lr > 0).sum(axis=0)
high_risk_support_pct = lambda hr, lr: round((hr > 0).sum(axis=0).astype(float) / len(hr), 3)
low_risk_support_pct = lambda hr, lr: round((lr > 0).sum(axis=0).astype(float) / len(lr), 3)

def hr_lr_ttest(hr, lr):
    """Returns the t-test (T statistic and p value), comparing the features for
    high- and low-risk entities."""
    res = stats.ttest_ind(
        hr.to_numpy(), lr.to_numpy(), axis=0, nan_policy="omit", equal_var=False
    )

    r0 = pd.Series(res[0], index=hr.columns)
    r1 = pd.Series(res[1], index=hr.columns)

    return pd.DataFrame({"ttest_T": r0, "ttest_p": r1})


# the crosstab functions are passed as a list of tuples;
# the first tuple element gives the name (or names) of the
# metric that the function calculates
crosstab_functions = [
    ("count_predicted_positive", high_risk_n),
    ("count_predicted_negative", low_risk_n),
    ("mean_predicted_positive", high_risk_mean),
    ("mean_predicted_negative", low_risk_mean),
    ("std_predicted_positive", high_risk_std),
    ("std_predicted_negative", low_risk_std),
    ("ratio_predicted_positive_over_predicted_negative", hr_lr_ratio),
    (["ttest_T", "ttest_p"], hr_lr_ttest),
]


def list_to_str(list_sql):
    str = "(" + ",".join(list_sql) + ")"
    return str


def populate_crosstabs_table(
    model_id,
    as_of_date,
    df,
    thresholds=None,
    crosstab_functions=crosstab_functions,
    push_to_db=True,
    return_df=False,
    engine=None,
    schema="results",
    table="crosstabs",
):
    """This function populates the results.crosstabs table.

    Args:
        entity_id_list (list): A list of entity_ids. Crosstabs will
                               be calculated using only predictions and
                               features of these entities. This list is
                               being passed to get_predictions_query and
                               get_features_query.
        crosstab_functions (dict): A dictionary of function names and
                               functions. The functions receive the feature
                               dataframes for entities that were predicted
                               postive or negative, respectively. Each function
                               returns a Series or DataFrame with rows now
                               indexed by the feature names, and columns
                               corresponding to the statistics that the
                               function calculated.
        thresholds (dict): A dictionary that maps column names to lists of
                           threshold values. The (name, threshold) pairs will
                           be applied to the dataframes with predictions to
                           split low- and high-risk entities.
        push_to_db (bool): If True, the crosstab table is being populated.
        return_df (bool): If True, a dataframe corresponding to the crosstabe
                          table will be returned.
        engine (SQLAlchemy engine): DB connection
        schema (str): Name of the schema to store results; defaults to 'results'
        table (str): Name of table to store results; defaults to 'crosstabs'

        Returns: A DataFrame, corresponding to results.crosstabs
    """
    logger.info(
        f"RUNNING populate_crosstabs_table for model_id {model_id} and as_of_date {as_of_date}",
    )
    if len(df) == 0:
        raise ValueError("No data could be fetched.")

    dfs = []

    # this will be useful later - the non-feature columns we'll grab
    model_cols = [
        "model_id",
        "as_of_date",
        "entity_id",
        "score",
        "rank_abs",
        "rank_pct",
        "label_value",
    ]

    # drop the all-null columns
    null_cols = df.columns[df.isnull().sum() == df.shape[0]]
    null_cols = [c for c in null_cols if c not in model_cols]

    if len(null_cols) > 0:
        logger.warning("Dropping the all-null columns: %s" % str(null_cols))
        df = df.drop(null_cols, 1)

    # if the dataframe isn't dummified, do it
    if object in df.dtypes.values or pd.Categorical.dtype in df.dtypes.values:
        to_dummify = [
            c
            for c in df.select_dtypes(include=["category", object])
            if c not in model_cols
        ]
        logger.debug("Dummifying the data")
        df = pd.get_dummies(df, columns=to_dummify, dummy_na=True)

    feat_cols = [c for c in df.columns if c not in model_cols]
    logger.debug("Iterating over thresholds to generate results...")
    for thres_type, thress in thresholds.items():
        for thres in thress:

            results = pd.DataFrame(index=feat_cols)
            logger.debug("split dataframe in high risk and lowriks")
            # split dataframe into high/low risk
            df_pred_pos = df.loc[df[thres_type] <= thres, feat_cols]
            df_pred_neg = df.loc[df[thres_type] > thres, feat_cols]
            logger.debug(f"len of hr: {len(df_pred_pos)} and lr: {len(df_pred_neg)}")
            for name, func in crosstab_functions:
                logger.debug(name)
                this_result = pd.DataFrame(func(df_pred_pos, df_pred_neg))
                if name in ["ttest_T", "ttest_p"]:
                    logger.debug("this_result:", this_result.shape)
                    logger.debug("Results:", results.shape)
                if not type(name) in [list, tuple]:
                    name = [name]
                # the metric name is coming from the crosstab_functions tuples
                this_result.columns = name
                results = results.join(this_result, how="outer")

            results = results.stack()
            results.index.names = ["feature_column", "metric"]
            results.name = "value"
            results = results.to_frame()

            results["model_id"] = model_id
            results["as_of_date"] = as_of_date
            results["threshold_unit"] = thres_type[-3:]
            results["threshold_value"] = thres

            dfs.append(results)

    df = pd.concat(dfs)
    if push_to_db:
        logger.debug("Pushing results to database...")
        df.reset_index().set_index(["model_id", "as_of_date", "metric"]).pg_copy_to(
            schema=schema, name=table, con=engine, if_exists="append"
        )
    if return_df:
        return df


def run_crosstabs(db_engine, crosstabs_config):
    crosstabs_query = """
        with models_list_query as (
        {models_list_query}
        ), as_of_dates_query as (
        {as_of_dates_query}
        ),models_dates_join_query as (
        {models_dates_join_query}
        ),features_query as (
        {features_query}
        ), predictions_query as (
        {predictions_query}
        )
        select * from predictions_query
        left join features_query f using (model_id,entity_id, as_of_date)
        """.format(
        models_list_query=crosstabs_config.models_list_query,
        as_of_dates_query=crosstabs_config.as_of_dates_query,
        models_dates_join_query=crosstabs_config.models_dates_join_query,
        features_query=crosstabs_config.features_query,
        predictions_query=crosstabs_config.predictions_query,
    )
    if len(crosstabs_config.entity_id_list) > 0:
        crosstabs_query += " where entity_id=ANY('{%s}') " % ", ".join(
            map(str, crosstabs_config.entity_id_list)
        )
    crosstabs_query += "  order by model_id, as_of_date, rank_abs asc;"
    df = pd.read_sql(crosstabs_query, db_engine)
    if len(df) == 0:
        raise ValueError("No data could be fetched.")
    groupby_obj = df.groupby(["model_id", "as_of_date"])
    for group, values in groupby_obj:
        df_modelid_asofdate = groupby_obj.get_group(group)
        res = populate_crosstabs_table(
            model_id=group[0],
            as_of_date=group[1],
            df=df_modelid_asofdate,
            thresholds=crosstabs_config.thresholds,
            crosstab_functions=crosstab_functions,
            push_to_db=True,
            return_df=False,
            engine=db_engine,
            schema=crosstabs_config.output["schema"],
            table=crosstabs_config.output["table"],
        )


def run_crosstabs_from_matrix(db_engine, project_path, model_id, threshold_type, threshold, matrix_uuid=None, push_to_db=True, table_schema='test_results', table_name='crosstabs', return_as_dataframe=True, replace=False):
    """ Calculate crosstabs for a model based on the matrix. 
        
        Args: 
            db_engine: Database engine
            project_path (str): Path where the experiment artifacts (models and matrices) are stored
            thresholds (Dict{str: Union[float, int}]): A dictionary that maps threhold type to the threshold
                                                    The threshold type can be one of the rank columns in the test_results.predictions_table
           
            matrix_uuid (str, optional): To run crosstabs for a different matrix than the validation matrix from the experiment

            push_to_db (bool, optional): Whether to write the results to the database. Defaults to True
            
            table_schame (str, optional): Database schema to store the crosstabs table. Defaults to `test_results`
            table_name (str, optional): Table name to use. Defaults to `crosstabs`. If the table exists, results are appended

        return:
            Dataframe of crosstabs
    """
    
    logging.info('Fetching predictions')
    
    where_clause = f'where model_id = {model_id}'
    
    if matrix_uuid:
        where_clause += f" and matrix_uuid = '{matrix_uuid}'"
    
    q = f'''
        select 
        matrix_uuid,
        entity_id,
        as_of_date,
        score, 
        label_value,
        test_label_timespan,
        {threshold_type},
        case when {threshold_type} <= {threshold} then 1 else 0 end as high_risk
        from test_results.predictions p
        {where_clause}
    '''
    
    id_columns = ['entity_id', 'as_of_date']
    
    predictions = pd.read_sql(q, db_engine).set_index(id_columns)
    
    if predictions.empty:
        logging.error('No predictions were found for the model and matrix')
        return 

    # The same model can have multiple test matrices. If not specified, we calculate crosstabs for all matrices
    matrices = [matrix_uuid]
    if matrix_uuid is None:
        matrices = predictions.matrix_uuid.unique()
        
    logging.info(f'Matrices for crosstabs: {matrices}')
    
    crosstab_tasks = list() # store a list of matrix uuids we need crosstabs for
        
    if table_exists(f'{table_schema}.{table_name}', db_engine):
        logging.info(f'The {table_schema}.{table_name} table exists. Checking for existing crosstabs')
        
        for matrix_uuid in matrices: 
            df = pd.read_sql(f'''
                    select 1 from {table_schema}.{table_name}
                    where model_id = {model_id}
                    and matrix_uuid = '{matrix_uuid}'
                    and threshold_type = '{threshold_type}'
                    and threshold = {threshold}
                ''', db_engine)
            
            if not df.empty:
                if replace: 
                    logging.info(f'Exsiting crosstabs found for model {model_id} and matrix {matrix_uuid}. Replace is True. Deleting.')
                    q = f'''
                        delete from {table_schema}.{table_name}
                        where model_id = {model_id}
                        and matrix_uuid = '{matrix_uuid}'
                        and threshold_type = '{threshold_type}'
                        and threshold = {threshold}
                    '''
                    
                    db_engine.execute(q)
                                        
                else:
                    logging.info(f'Existing crosstabs found for model {model_id} and matrix {matrix_uuid}. Replace flag is not set. Skipping')
                    continue
            
            logging.info(f'Crosstabs task added for matrix {matrix_uuid}')
            crosstab_tasks.append(matrix_uuid)
            
    else:
        logging.info(f'{table_schema}.{table_name} does not exist. Calculating crostabs for all matrices')
        crosstab_tasks = matrices
            
    if len(crosstab_tasks) == 0:
        logging.warning('Crosstabs calculation not needed. Exiting.')
        return
    
    project_storage = ProjectStorage(project_path)
    matrix_storage_engine = project_storage.matrix_storage_engine()
    
    results = list()
    for m in crosstab_tasks:
        matrix = matrix_storage_engine.get_store(matrix_uuid=m).design_matrix
        
        feature_names = matrix.columns
        
        matrix = matrix.join(predictions[predictions['matrix_uuid'] == m], how='left')
        
        # topk_msk = matrix[threshold_type] <= threshold
        topk_msk = matrix['high_risk'] == 1
        
        positives = matrix[topk_msk][feature_names]
        negatives = matrix[~topk_msk][feature_names]
        
        for name, function in crosstab_functions:
            logging.info(f'Calculating {name}')
            
            if name == ["ttest_T", "ttest_p"]:
                res = function(positives, negatives)

                this_result = res.drop(columns=['ttest_p'])
                
                # TODO: This is very hacky and ugly
                this_result.rename(columns={'ttest_T': 0}, inplace=True)
                this_result['metric'] = 'ttest_T'
                results.append(this_result)
                
                this_result = res.drop(columns=['ttest_T'])
                
                # TODO: This is very hacky and ugly
                this_result.rename(columns={'ttest_p': 0}, inplace=True)
                this_result['metric'] = 'ttest_p'
                
                results.append(this_result)
            
                continue
                
            this_result = pd.DataFrame(function(positives, negatives))
            
            this_result['metric'] = name
                
            results.append(this_result)
                        
    results = pd.concat(results).reset_index()
    results.rename(columns={'index': 'feature', 0:'value'}, inplace=True)
    results['threshold_type'] = threshold_type
    results['threshold'] = threshold
    results['model_id'] = model_id
    results['matrix_uuid'] = matrix_uuid
    
    if push_to_db:
        logging.info(f'Pushing the results to the database, {len(results)} rows')
                
        results.set_index(
            ['model_id', 'matrix_uuid', 'feature', 'metric', 'threshold_type', 'threshold'],
            inplace=True
        )
        
        results = results.reset_index()
        
        if not table_exists(f'{table_schema}.{table_name}', db_engine):
            q = f'''
                create schema if not exists {table_schema};
                
                create table {table_schema}.{table_name} (
                  model_id INTEGER,
                  matrix_uuid TEXT,
                  feature TEXT,
                  metric TEXT,
                  threshold_type TEXT,
                  threshold FLOAT,
                  value FLOAT  
                );
            
            '''
            # q = _generate_create_table_sql_statement_from_df(results, f'{table_schema}.{table_name}')
            db_engine.execute(q)
        
        conn = db_engine.raw_connection()
        cursor = conn.cursor()
        
        buffer = StringIO()
        results.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        
        columns = ', '.join(results.columns)
        print(columns)
        cursor.copy_expert(f"COPY {table_schema}.{table_name} ({columns}) FROM STDIN WITH CSV", buffer)
        # results.to_sql(con=db_engine, schema=table_schema, name=table_name, if_exists='append')
        conn.commit()
        cursor.close()
        conn.close()
        
    return results
        
                
        