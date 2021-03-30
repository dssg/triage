import pandas as pd

def get_aequitas_results(engine, parameter, schema="test_results", table="aequitas", model_id=None,  subset_hash="", tie_breaker="worst"):
    ''' This function returns the current contents of the aequitas table.

        Args:
            - engine: SQLAlchemy engine conected to database
            - parameter: A string that indicates any parameters for the metric (ex. `100_abs` indicates top-100 entities)
            - schema: Databse schema to find table within
            - table: Databse table to select data from
            - model_id: A model_id, to query only for results of that model
            - subset_hash: Identifies the subset for the evaluation
            - tie_breaker: Indicates how ties are broken

        Returns: A DataFrame, corresponding to schema.table
    '''

    query = f"""SELECT * FROM {schema}.{table} 
                 WHERE parameter = '{parameter}' 
                   AND subset_hash = '{subset_hash}'
                   AND tie_breaker = '{tie_breaker}'
                   """
    if model_id:
        query += f" AND model_id = {model_id}"
    return pd.read_sql(query, con=engine)