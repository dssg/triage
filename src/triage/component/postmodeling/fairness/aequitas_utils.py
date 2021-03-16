import pandas as pd

def get_aequitas_results(engine, schema="test_results", table="aequitas", model_id=None):
    ''' This function returns the current contents of the aequitas table.

        Args:
            - engine: SQLAlchemy engine conected to database
            - schema: Databse schema to find table within
            - table: Databse table to select data from
            - model_id: A model_id, to query only for results of that model

        Returns: A DataFrame, corresponding to schema.table
    '''

    query = f"SELECT * FROM {schema}.{table}"
    if model_id:
        query += f" WHERE model_id = {model_id}"
    return pd.read_sql(query, con=engine)