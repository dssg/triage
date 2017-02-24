import logging


def get_feature_names(table_name, entity_column, schema_name, engine):
    fname = 'feature_names.csv'
    feature_names_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table}' AND
              table_schema = '{schema}' AND
              column_name NOT IN ('{entity}', 'date')
    """.format(table=table_name, schema=schema_name, entity=entity_column)
    logging.debug(feature_names_query)
    write_to_csv(feature_names_query, fname, engine, '')
    with open(fname) as f:
        feature_names = f.readlines()
    return [name.strip() for name in feature_names]


def add_imputation(feature_names):
    return [
        """,
               CASE
                   WHEN f.{0} IS NULL THEN 0
                   ELSE f.{0}
               END as {0}""".format(name) for name in feature_names
    ]


def build_feature_query(table_name, feature_names, schema, feature_dates, engine):
    """ Given a table, schema, and list of dates, write a query to perform a
    left outer join on the entity date table
    :param table_name: feature table to query
    :param schema: name of the feature schema
    :param feature_dates: dates to query for
    :type table_name: str
    :type schema: str
    :type feature_dates: list
    :return: query for feature table
    :rtype: str
    """
    feature_dates_tuple = tuple([str(date) for date in feature_dates])
    feature_names_with_imputation = add_imputation(feature_names)

    query = """
        SELECT ed.entity_id,
               ed.as_of_date{features}
        FROM {schema_name}.tmp_entity_date ed
        LEFT OUTER JOIN {schema_name}.{feature_table} f
        ON ed.entity_id = f.entity_id AND
           ed.as_of_date = f.as_of_date AND
           ed.as_of_date in {date_list}
        ORDER BY ed.entity_id,
                 ed.as_of_date
    """.format(
        features=''.join(feature_names_with_imputation),
        schema_name=schema,
        date_list=feature_dates_tuple,
        feature_table=table_name
    )
    return(query)


def write_to_csv(query_string, file_name, engine, header='HEADER'):
    """ Given a query, write the requested data to csv.
    :param query_string: query to send
    :param file_name: name to save the file as
    :param engine: database connection
    :header: text to include in query indicating if a header should be saved in
             output
    :type query_string: str
    :type file_name: str
    :type engine: psycopg2 engine
    :type header: str
    :return: none
    :rtype: none
    """
    matrix_csv = open(file_name, 'wb')
    copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
        query=query_string,
        head=header
    )
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.copy_expert(copy_sql, matrix_csv)


def make_entity_dates_table(engine, feature_dates, feature_tables, schema):
    """ Make a table containing the entity_ids and feature dates required for
    the current matrix.
    :param engine: a postgresql database engine
    :param feature_dates: the dates required
    :param feature_tables: the tables to be used for the current matrix
    :param schema: the name of the features schema
    :type engine: sqlalchemy engine
    :type feature_dates: list
    :type feature_tables: list
    :type schema: str
    :return: none
    :rtype: none
    """
    feature_dates_tuple = tuple([str(date) for date in feature_dates])
    query_list = []
    for table in feature_tables:
        union = ''
        if feature_tables.index(table) != 0:
            union = 'UNION'
        subquery = """ {u}
            SELECT DISTINCT entity_id, as_of_date
            FROM {schema_name}.{table_name}
            WHERE as_of_date IN {dates}
        """.format(
            u=union,
            table_name=table,
            dates=feature_dates_tuple,
            schema_name=schema
        )
        query_list.append(subquery)
    query = """
        CREATE TABLE {schema_name}.tmp_entity_date
        AS ({subqueries})
    """.format(schema_name=schema, subqueries=''.join(query_list))
    engine.execute(query)
