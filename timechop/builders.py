import io
import logging
import pandas
from metta import metta_io as metta
import os
import csv


class BuilderBase(object):
    def __init__(self, db_config, matrix_directory, engine, replace=True):
        self.db_config = db_config
        self.matrix_directory = matrix_directory
        self.engine = engine
        self.replace = replace

    def build_all_matrices(self, build_tasks):
        logging.info('Building %s matrices', len(build_tasks.keys()))
        for matrix_uuid, task_arguments in build_tasks.items():
            self.build_matrix(**task_arguments)

    def _format_imputations(self, feature_names):
        """ For a list of feature columns, format them for a SQL query, imputing
        0 for missing values.

        :param feature_names: names of feature columns in a single table
        :type feature_names: list

        :return: strings to add to feature query to select features and make
                 imputations
        :rtype: list
        """
        imputations = [
            """,
                    CASE
                        WHEN "{0}" IS NULL THEN 0
                        ELSE "{0}"
                    END as "{0}" """.format(feature_names) for feature_names in feature_names
        ]
        return(imputations)

    def build_labels_query(
            self,
            as_of_times,
            final_column,
            label_name,
            label_type,
            prediction_window
        ):
        """ Given a table, schema, and list of dates, write a query to get the
        list of valid as_of_date-entity pairs, and, if requested, the labels.

        :param final_column: string to add to the end of select clause; empty if
                             only entity_ids and as_of_dates are desired; comma
                             plus name of label values column if labels are
                             desired
        :type labels_table_name: list
        :type final_column: str

        :return: query for labels table
        :rtype: str
        """
        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]
        query = """
            SELECT entity_id,
                   as_of_date{labels}
            FROM {labels_schema_name}.{labels_table_name}
            WHERE as_of_date IN (SELECT (UNNEST (ARRAY{times}::timestamp[]))) AND
                  label_name = '{l_name}' AND
                  label_type = '{l_type}' AND
                  prediction_window = '{window}'
            ORDER BY entity_id,
                     as_of_date
        """.format(
            labels = final_column,
            labels_schema_name = self.db_config['labels_schema_name'],
            labels_table_name = self.db_config['labels_table_name'],
            times = as_of_time_strings,
            l_name = label_name,
            l_type = label_type,
            window = prediction_window
        )
        return(query)

    def build_outer_join_query(
        self,
        as_of_times,
        right_table_name,
        right_column_selections,
        entity_date_table_name,
        additional_conditions = ''
    ):
        """ Given a (features or labels) table, a list of times, columns to
        select, and (optionally) a set of join conditions, perform an outer
        join to the entity date table.

        :param as_of_times: the times to include in the join
        :param right_table_name: the name of the right (feature/label) table
        :param right_column_selections: formatted text for the columns to select
        :param entity_date_table_name: name of table containing all valid entity ids and dates
        :param additional_conditions: formatted text for additional join
                                      conditions
        :type as_of_times: list
        :type right_table_name: str
        :type right_column_selections: str
        :type entity_date_table_name: str
        :type additional_conditions: str

        :return: postgresql query for the outer join to the entity-dates table
        :rtype: str
        """
        # format inputs for adding to query
        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]

        # put everything into the query
        query = """
            SELECT ed.entity_id,
                   ed.as_of_date{columns}
            FROM {entity_date_table_name} ed
            LEFT OUTER JOIN {right_table} r
            ON ed.entity_id = r.entity_id AND
               ed.as_of_date = r.as_of_date AND
               ed.as_of_date IN (SELECT (UNNEST (ARRAY{times}::timestamp[]))){more}
            ORDER BY ed.entity_id,
                     ed.as_of_date
        """.format(
            columns = ''.join(right_column_selections),
            feature_schema = self.db_config['features_schema_name'],
            entity_date_table_name = entity_date_table_name,
            right_table = right_table_name,
            times = as_of_time_strings,
            more = additional_conditions
        )
        return(query)


    def make_entity_date_table(
        self,
        as_of_times,
        label_name,
        label_type,
        feature_table_names,
        matrix_type,
        matrix_uuid,
        prediction_window
    ):
        """ Make a table containing the entity_ids and as_of_dates required for
        the current matrix.

        :param feature_tables: the tables to be used for the current matrix
        :param schema: the name of the features schema
        :type feature_tables: list
        :type schema: str

        :return: table name
        :rtype: str
        """
        if matrix_type == 'train':
            indices_query = self.build_labels_query(
                as_of_times=as_of_times,
                final_column='',
                label_name=label_name,
                label_type=label_type,
                prediction_window=prediction_window
            )
        elif matrix_type == 'test':
            indices_query = self.get_all_valid_entity_date_combos(
                as_of_times=as_of_times,
                feature_table_names=feature_table_names
            )
        else:
            raise ValueError('Unknown matrix type passed: {}'.format(matrix_type))

        table_name = '_'.join([matrix_uuid, 'tmp_entity_date'])
        query = """
            DROP TABLE IF EXISTS {features_schema_name}."{table_name}";
            CREATE TABLE {features_schema_name}."{table_name}"
            AS ({index_query})
        """.format(
            features_schema_name=self.db_config['features_schema_name'],
            table_name=table_name,
            index_query=indices_query
        )
        self.engine.execute(query)

        return table_name

    def get_all_valid_entity_date_combos(self, as_of_times, feature_table_names):
        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]     
        query_list = []       
        for index, table in enumerate(feature_table_names):
            union = ''        
            if index != 0:      
                union = 'UNION'       
            subquery = """ {u}        
                SELECT DISTINCT entity_id, as_of_date     
                FROM {schema_name}.{table_name}       
                WHERE as_of_date IN (SELECT (UNNEST (ARRAY{dates}::timestamp[])))       
            """.format(       
                u = union,        
                table_name = table,       
                dates = as_of_time_strings,      
                schema_name = self.db_config['features_schema_name']      
            )     
            query_list.append(subquery)
        
        return(''.join(query_list))


class CSVBuilder(BuilderBase):
    def build_matrix(
        self,
        as_of_times,
        label_name,
        label_type,
        feature_dictionary,
        matrix_directory,
        matrix_metadata,
        matrix_uuid,
        matrix_type
    ):
        """ Write a design matrix to disk with the specified paramters.

        :param as_of_times: datetimes to be included in the matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param feature_dictionary: a dictionary of feature tables and features
                                   to be included in the matrix
        :param matrix_directory: the directory in which to store the matrix
        :param matrix_metadata: a dictionary of metadata about the matrix
        :param matrix_uuid: a unique id for the matrix
        :param matrix_type: the type (train/test) of matrix
        :type as_of_times: list
        :type label_name: str
        :type label_type: str
        :type feature_dictionary: dict
        :type matrix_directory: str
        :type matrix_metadata: dict
        :type matrix_uuid: str
        :type matrix_type: str

        :return: none
        :rtype: none
        """
        matrix_filename = os.path.join(
            matrix_directory,
            '{}.csv'.format(matrix_uuid)
        )
        if not self.replace and os.path.exists(matrix_filename):
            logging.info('Skipping %s because matrix already exists', matrix_filename)
            return

        logging.info('Creating matrix %s > %s', matrix_metadata['matrix_id'], matrix_filename)
        # make the entity time table and query the labels and features tables
        logging.info('Making entity date table')
        entity_date_table_name = self.make_entity_date_table(
            as_of_times,
            label_name,
            label_type,
            feature_dictionary.keys(),
            matrix_type,
            matrix_uuid,
            matrix_metadata['prediction_window']
        )
        try:
            logging.info('Writing feature group data')
            features_csv_names = self.write_features_data(
                as_of_times,
                feature_dictionary,
                entity_date_table_name,
                matrix_uuid
            )
            try:
                logging.info('Writing label data')
                labels_csv_name = self.write_labels_data(
                    as_of_times,
                    label_name,
                    label_type,
                    matrix_type,
                    entity_date_table_name,
                    matrix_uuid,
                    matrix_metadata['prediction_window']
                )
                features_csv_names.insert(0, labels_csv_name)

                # stitch together the csvs
                logging.info('Merging features data')
                output = self.merge_feature_csvs(
                    features_csv_names,
                    matrix_directory,
                    matrix_uuid
                )
            finally:
                # clean up files and database before finishing
                for csv_name in features_csv_names:
                    self.remove_file(csv_name)
            try:
                # store the matrix
                logging.info('Archiving matrix with metta')
                metta.archive_matrix(
                    matrix_config=matrix_metadata,
                    df_matrix=output,
                    overwrite=True,
                    directory=self.matrix_directory,
                    format='csv'
                )
            finally:
                if isinstance(output, str):
                    os.remove(output)
        finally:
            self.engine.execute(
                'drop table "{}"."{}";'.format(
                    self.db_config['features_schema_name'],
                    entity_date_table_name
                )
            )


    def write_labels_data(
        self,
        as_of_times,
        label_name,
        label_type,
        matrix_type,
        entity_date_table_name,
        matrix_uuid,
        prediction_window
    ):
        """ Query the labels table and write the data to disk in csv format.
        
        :return: name of csv containing labels
        :rtype: str
        """
        csv_name = os.path.join(
            self.matrix_directory,
            '{}-{}.csv'.format(matrix_uuid, self.db_config['labels_table_name'])
        )
        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]
        if matrix_type == 'train':
            labels_query = self.build_labels_query(
                as_of_times=as_of_times,
                final_column=', label as {}'.format(label_name),
                label_name=label_name,
                label_type=label_type,
                prediction_window=prediction_window
            )
        elif matrix_type == 'test':
            labels_query=self.build_outer_join_query(
                as_of_times=as_of_times,
                right_table_name='{schema}.{table}'.format(
                    schema=self.db_config['labels_schema_name'],
                    table=self.db_config['labels_table_name']
                ),
                entity_date_table_name='"{schema}"."{table}"'.format(
                    schema=self.db_config['features_schema_name'],
                    table=entity_date_table_name
                ),
                right_column_selections=', r.label as {}'.format(label_name),
                additional_conditions='''AND
                    r.label_name = '{name}' AND
                    r.label_type = '{type}' AND
                    r.prediction_window = '{window}'
                '''.format(
                    name=label_name,
                    type=label_type,
                    window=prediction_window
                )
            )
        else:
            raise ValueError(
                'Did not recognize matrix type: {}'.format(matrix_type)
            )
        self.write_to_csv(labels_query, csv_name)
        return(csv_name)

    def write_features_data(self, as_of_times, feature_dictionary, entity_date_table_name, matrix_uuid):
        """ Loop over tables in features schema, writing the data from each to a
        csv. Return the full list of feature csv names and the list of all
        features.

        :return: list of csvs containing feature data
        :rtype: tuple
        """
        # iterate! for each table, make query, write csv, save feature & file names
        features_csv_names = []
        for feature_table_name, feature_names in feature_dictionary.items():
            logging.info('Retrieving feature data from %s', feature_table_name)
            csv_name = os.path.join(
                self.matrix_directory,
                '{}-{}.csv'.format(matrix_uuid, feature_table_name)
            )
            features_query = self.build_outer_join_query(
                as_of_times = as_of_times,
                right_table_name = '{schema}.{table}'.format(
                    schema = self.db_config['features_schema_name'],
                    table = feature_table_name
                ),
                entity_date_table_name = '{schema}."{table}"'.format(
                    schema = self.db_config['features_schema_name'],
                    table = entity_date_table_name
                ),
                right_column_selections = self._format_imputations(feature_names)
            )
            self.write_to_csv(features_query, csv_name)
            features_csv_names.append(csv_name)

        return(features_csv_names)

    def write_to_csv(self, query_string, file_name, header = 'HEADER'):
        """ Given a query, write the requested data to csv.

        :param query_string: query to send
        :param file_name: name to save the file as
        :header: text to include in query indicating if a header should be saved
                 in output
        :type query_string: str
        :type file_name: str
        :type header: str

        :return: none
        :rtype: none
        """
        matrix_csv = self.open_fh_for_writing(file_name)
        try:
            copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
                query = query_string,
                head = header
            )
            conn = self.engine.raw_connection()
            cur = conn.cursor()
            cur.copy_expert(copy_sql, matrix_csv)
        finally:
            self.close_filehandle(file_name)


class LowMemoryCSVBuilder(CSVBuilder):
    def __init__(self, *args, **kwargs):
        super(LowMemoryCSVBuilder, self).__init__(*args, **kwargs)
        self.filehandles = {}

    def open_fh_for_writing(self, filename):
        self.filehandles[filename] = open(filename, 'wb')
        return self.filehandles[filename]

    def open_fh_for_reading(self, filename):
        return open(filename)

    def close_filehandles(self):
        for fh in self.filehandles.values():
            fh.close()

    def close_filehandle(self, filename):
        self.filehandles[filename].close()

    def remove_file(self, filename):
        del self.filehandles[filename]
        os.remove(filename)

    def merge_feature_csvs(self, source_filenames, matrix_directory, matrix_uuid):
        """Horizontally merge a list of feature CSVs
        Assumptions:
        - The first and second columns of each CSV are
          the entity_id and date
        - That the CSVs have the same list of entity_id/date combinations
          in the same order.
        - The first CSV is expected to be labels, and only have
          entity_id, date, and label.
        - All other CSVs do not have any labels (all non entity_id/date columns
          will be treated as features)
        - The label will be in the *last* column of the merged CSV

        :param source_filenames: the filenames of each feature csv
        :param matrix_directory: the directory that the final CSV will reside in
        :param matrix_uuid: the uuid of the final matrix
        :type source_filenames: list
        :type matrix_directory: str
        :type matrix_uuid: str

        :return: none
        :rtype: none

        :raises: ValueError if the first two columns in every CSV don't match
        """
        temp_matrix_filename = os.path.join(
            matrix_directory,
            'tmp_{}.csv'.format(matrix_uuid)
        )
        with open(temp_matrix_filename, 'w') as outfile:
            writer = csv.writer(outfile)
            source_filehandles = [self.open_fh_for_reading(fname) for fname in source_filenames]
            source_readers = [csv.reader(fh) for fh in source_filehandles]
            try:
                headers = None
                for rows in zip(*source_readers):
                    if not headers:
                        labels_table_header, other_table_headers = rows[0], rows[1:]
                        entity_id, date, label = labels_table_header
                        headers = [entity_id, date] + [
                            column
                            for sublist in other_table_headers
                            for column in sublist[2:]
                            if column not in labels_table_header
                        ] + [label]
                        writer.writerow(headers)
                        logging.debug('Found headers: %s', headers)
                        logging.info('Found {} headers'.format(len(headers)))
                        continue
                    entity_ids = []
                    dates = []
                    all_features = []
                    label_observed = None
                    features = []
                    for row in rows:
                        if not label_observed:
                            if len(row) == 3:
                                entity_id, date, label = row
                            elif len(row) == 2:
                                entity_id, date = row
                                label = None
                            else:
                                raise ValueError('''
                                    Unexpected number of values observed in 
                                    labels: {}
                                '''.format(row))
                            label_observed = True
                        else:
                            entity_id, date, features = row[0], row[1], row[2:]
                        entity_ids.append(entity_id)
                        dates.append(date)
                        if not isinstance(features, list):
                            features = [features]
                        all_features += features
                    entity_ids = list(set(entity_ids))
                    dates = list(set(dates))
                    if len(entity_ids) > 1 or len(dates) > 1:
                        raise ValueError('''
                        Either multiple entity ids or dates
                        found in parallel rows. entity_ids: %s dates: %s
                        ''', entity_ids, dates)
                    writer.writerow(entity_ids + dates + all_features + [label])
            finally:
                self.close_filehandles()
        return temp_matrix_filename


class HighMemoryCSVBuilder(CSVBuilder):
    def __init__(self, *args, **kwargs):
        super(HighMemoryCSVBuilder, self).__init__(*args, **kwargs)
        self.filehandles = {}

    def open_fh_for_writing(self, filename):
        self.filehandles[filename] = io.StringIO()
        return self.filehandles[filename]

    def open_fh_for_reading(self, filename):
        self.filehandles[filename].seek(0)
        return self.filehandles[filename]

    def close_filehandles(self):
        pass

    def close_filehandle(self, filename):
        pass

    def remove_file(self, filename):
        del self.filehandles[filename]

    def merge_feature_csvs(self, source_filenames, matrix_directory, matrix_uuid):
        """Horizontally merge a list of feature CSVs
        Assumptions:
        - The first and second columns of each CSV are
          the entity_id and date
        - That the CSVs have the same list of entity_id/date combinations
          in the same order.
        - The first CSV is expected to be labels, and only have
          entity_id, date, and label.
        - All other CSVs do not have any labels (all non entity_id/date columns
          will be treated as features)
        - The label will be in the *last* column of the merged CSV

        :param source_filenames: the filenames of each feature csv
        :param out_filename: the desired filename of the merged csv
        :type source_filenames: list
        :type out_filename: str

        :return: none
        :rtype: none

        :raises: ValueError if the first two columns in every CSV don't match
        """

        source_filehandles = [self.open_fh_for_reading(fname) for fname in source_filenames]
        dataframes = []
        for filehandle in source_filehandles:
            df = pandas.read_csv(filehandle)
            df.set_index(['entity_id', 'as_of_date'], inplace=True)
            dataframes.append(df)

        big_df = dataframes[1].join(dataframes[2:] + [dataframes[0]])
        return big_df
