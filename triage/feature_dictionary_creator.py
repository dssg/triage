
class FeatureDictionaryCreator(object):
    def __init__(self, features_schema_name, db_engine):
        self.features_schema_name = features_schema_name
        self.db_engine = db_engine

    def feature_dictionary(self, tables_to_exclude):
        """ Create a dictionary of feature names, where keys are feature tables
        and values are lists of feature names.

        :return: feature_dictionary
        :rtype: dict
        """
        # prepare for iteration! get items to iterate over & initialize results
        feature_table_names = [
            row[0] for row in
            self.db_engine.execute(self._build_feature_tables_list_query())
            if row[0] not in tables_to_exclude
        ]
        feature_dictionary = {}

        # iterate! store each table name + features names as key-value pair
        for feature_table_name in feature_table_names:
            feature_names = [
                row[0] for row in
                self.db_engine.execute(
                    self._build_feature_names_query(feature_table_name)
                )
            ]
            feature_dictionary[feature_table_name] = feature_names
        return(feature_dictionary)

    def _build_feature_tables_list_query(self):
        """ Write a query to get a list of tables in the feature schema.

        :return: query
        :rtype: str
        """
        # format the query that gets table names,
        feature_table_names_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
        """.format(
            schema=self.features_schema_name
        )

        return(feature_table_names_query)

    def _build_feature_names_query(self, table_name):
        """ For a given feature table, get the names of the feature columns.

        :param table_name: name of the feature table
        :type table_name: str

        :return: names of the feature columns in given table
        :rtype: list
        """
        # format the query that gets column names,
        # excluding indices from result
        feature_names_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}' AND
                  table_schema = '{schema}' AND
                  column_name NOT IN ('entity_id', 'as_of_date')
        """.format(
            table=table_name,
            schema=self.features_schema_name
        )

        return(feature_names_query)
