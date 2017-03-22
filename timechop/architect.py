import csv
import logging
import metta.metta_io as metta
import os
import itertools
from . import utils


class Architect(object):

    def __init__(self, beginning_of_time, label_names, label_types, db_config,
                 matrix_directory, user_metadata, engine):
        self.beginning_of_time = beginning_of_time # earliest time included in features
        self.label_names = label_names
        self.label_types = label_types
        self.db_config = db_config
        self.matrix_directory = matrix_directory
        self.user_metadata = user_metadata
        self.engine = engine

    def chop_data(self, matrix_set_definitions, feature_dictionary):
        updated_definitions = []
        completed_uuids = set()
        for matrix_set in matrix_set_definitions:
            for label_name, label_type in itertools.product(self.label_names, self.label_types):
                matrix_set['train_uuid'] = self.design_matrix(
                    matrix_definition = matrix_set['train_matrix'],
                    feature_dictionary = feature_dictionary,
                    label_name = label_name,
                    label_type = label_type,
                    completed_uuids = completed_uuids,
                    matrix_type = 'train'
                )
                completed_uuids.add(matrix_set['train_uuid'])
                test_uuids = []
                for test_matrix in matrix_set['test_matrices']:
                    test_uuid = self.design_matrix(
                        matrix_definition = test_matrix,
                        feature_dictionary = feature_dictionary,
                        label_name = label_name,
                        label_type = label_type,
                        completed_uuids = completed_uuids,
                        matrix_type = 'test'
                    )
                    test_uuids.append(test_uuid)
                    completed_uuids.add(test_uuid)
                matrix_set['test_uuids'] = test_uuids
                updated_definitions.append(matrix_set)

        return(updated_definitions)

    def design_matrix(
        self,
        matrix_definition,
        feature_dictionary,
        label_name,
        label_type,
        matrix_type,
        completed_uuids
    ):
        """ Generate matrix metadata and, if no such matrix has already been
        made this batch, build the matrix.

        :param end_time: the limit of the prediction window for the matrix
        :param feature_frequency: the time between feature & label as_of_dates
        :param label_name: the name of the label in the labels table
        :param label_type: the type of the label in the labels table
        :type end_time: datetime.datetime
        :type feature_frequency: str
        :type label_name: str
        :type label_type: str

        :return: uuid for the matrix
        :rtype: str
        """
        # make a human-readable label for this matrix
        matrix_id = '_'.join([
            label_name,
            label_type,
            str(matrix_definition['matrix_start_time']),
            str(matrix_definition['matrix_end_time'])
        ])

        # get a uuid
        matrix_metadata = self._make_metadata(
            matrix_definition,
            feature_dictionary,
            label_name,
            label_type,
            matrix_id,
            matrix_type
        )
        uuid = metta.generate_uuid(matrix_metadata)
        matrix_filename = os.path.join(
            self.matrix_directory,
            '{}.csv'.format(uuid)
        )

        if uuid not in completed_uuids:
            self.build_matrix(
                matrix_definition['as_of_times'],
                label_name,
                label_type,
                feature_dictionary,
                matrix_filename,
                matrix_metadata,
                matrix_type
            )

        return(uuid)

    def build_matrix(self, as_of_times, label_name, label_type,
                     feature_dictionary, matrix_filename, matrix_metadata,
                     matrix_type):
        """ Write a design matrix to disk with the specified paramters.

        :param as_of_dates: dates to be included in the matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param feature_dictionary: a dictionary of feature tables and features
                                   to be included in the matrix
        :param matrix_filename: the file name for the final matrix
        :param matrix_metadata: a dictionary of metadata about the matrix
        :type as_of_dates: list
        :type label_name: str
        :type label_type: str
        :type feature_dictionary: dict
        :type matrix_filename: str
        :type matrix_metadata: dict

        :return: none
        :rtype: none
        """
        # make the entity time table and query the labels and features tables
        self.make_entity_date_table(
            as_of_times,
            label_name,
            label_type,
            feature_dictionary.keys(),
            matrix_type
        )
        features_csv_names = self.write_features_data(
            as_of_times,
            feature_dictionary
        )
        labels_csv_name = self.write_labels_data(
            as_of_times,
            label_name,
            label_type,
            matrix_type
        )
        features_csv_names.insert(0, labels_csv_name)

        # stitch together the csvs
        self.merge_feature_csvs(features_csv_names, matrix_filename)

        # store the matrix
        metta.archive_matrix(matrix_metadata, matrix_filename, format = 'csv')

        # clean up files and database before finishing
        for csv_name in features_csv_names:
            os.remove(csv_name)
        self.engine.execute(
            'drop table {}.tmp_entity_date;'.format(self.db_config['features_schema_name'])
        )


    def write_labels_data(self, as_of_times, label_name, label_type,
                          matrix_type):
        """ Query the labels table and write the data to disk in csv format.
        
        :return: name of csv containing labels
        :rtype: str
        """
        csv_name = os.path.join(
            self.matrix_directory,
            '{}.csv'.format(self.db_config['labels_table_name'])
        )
        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]
        if matrix_type == 'train':
            labels_query = self.build_labels_query(
                as_of_times = as_of_times,
                final_column = ', label as {}'.format(label_name),
                label_name = label_name,
                label_type = label_type
            )
        elif matrix_type == 'test':
            labels_query = self.build_outer_join_query(
                as_of_times = as_of_times,
                right_table_name = '{schema}.{table}'.format(
                    schema = self.db_config['labels_schema_name'],
                    table = self.db_config['labels_table_name']
                ),
                right_column_selections = ', r.label as {}'.format(label_name),
                additional_conditions = '''AND
                    r.label_name = '{name}' AND
                    r.label_type = '{type}'
                '''.format(
                    name = label_name,
                    type = label_type
                )
            )
        else:
            raise ValueError(
                'Did not recognize matrix type: {}'.format(matrix_type)
            )
        self.write_to_csv(labels_query, csv_name)
        return(csv_name)

    def write_features_data(self, as_of_times, feature_dictionary):
        """ Loop over tables in features schema, writing the data from each to a
        csv. Return the full list of feature csv names and the list of all
        features.

        :return: list of csvs containing feature data
        :rtype: tuple
        """
        # iterate! for each table, make query, write csv, save feature & file names
        features_csv_names = []
        for feature_table_name, feature_names in feature_dictionary.items():
            csv_name = os.path.join(
                self.matrix_directory,
                '{}.csv'.format(feature_table_name)
            )
            features_query = self.build_outer_join_query(
                as_of_times = as_of_times,
                right_table_name = '{schema}.{table}'.format(
                    schema = self.db_config['features_schema_name'],
                    table = feature_table_name
                ),
                right_column_selections = self._format_imputations(feature_names)
            )
            self.write_to_csv(features_query, csv_name)
            features_csv_names.append(csv_name)

        return(features_csv_names)

    def _make_metadata(self, matrix_definition, feature_dictionary, label_name,
                       label_type, matrix_id, matrix_type):
        """ Generate dictionary of matrix metadata.

        :param feature_names: names of feature columns
        :param matrix_id: human-readable identifier for the matrix
        :type feature_names: list
        :type matrix_id: str

        :return: metadata needed for matrix identification and modeling
        :rtype: dict
        """
        matrix_metadata = {

            # temporal information
            'start_time': self.beginning_of_time,
            'end_time': matrix_definition['matrix_end_time'],

            # columns
            'indices': ['entity_id', 'as_of_date'],
            'feature_names': utils.feature_list(feature_dictionary),
            'label_name': label_name,

            # other information
            'label_type': label_type,
            'matrix_id': matrix_id,
            'matrix_type': matrix_type

        }
        matrix_metadata.update(matrix_definition)
        matrix_metadata.update(self.user_metadata)

        if 'prediction_window' not in matrix_definition.keys():
            matrix_metadata['prediction_window'] = '0d'

        return(matrix_metadata)

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

    def build_labels_query(self, as_of_times, final_column, label_name,
                           label_type):
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
                  label_type = '{l_type}'
            ORDER BY entity_id,
                     as_of_date
        """.format(
            labels = final_column,
            labels_schema_name = self.db_config['labels_schema_name'],
            labels_table_name = self.db_config['labels_table_name'],
            times = as_of_time_strings,
            l_name = label_name,
            l_type = label_type
        )
        return(query)

    def build_outer_join_query(
        self,
        as_of_times,
        right_table_name,
        right_column_selections,
        additional_conditions = ''
    ):
        """ Given a (features or labels) table, a list of times, columns to
        select, and (optionally) a set of join conditions, perform an outer
        join to the entity date table.

        :param as_of_times: the times to include in the join
        :param right_table_name: the name of the right (feature/label) table
        :param right_column_selections: formatted text for the columns to select
        :param additional_conditions: formatted text for additional join
                                      conditions
        :type as_of_times: list
        :type right_table_name: str
        :type right_column_selections: str
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
            FROM {feature_schema}.tmp_entity_date ed
            LEFT OUTER JOIN {right_table} r
            ON ed.entity_id = r.entity_id AND
               ed.as_of_date = r.as_of_date AND
               ed.as_of_date IN (SELECT (UNNEST (ARRAY{times}::timestamp[]))){more}
            ORDER BY ed.entity_id,
                     ed.as_of_date
        """.format(
            columns = ''.join(right_column_selections),
            feature_schema = self.db_config['features_schema_name'],
            right_table = right_table_name,
            times = as_of_time_strings,
            more = additional_conditions
        )
        return(query)

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
        matrix_csv = open(file_name,'wb')
        copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
            query = query_string,
            head = header
        )
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, matrix_csv)

    def make_entity_date_table(self, as_of_times, label_name, label_type,
                                feature_table_names, matrix_type):
        """ Make a table containing the entity_ids and as_of_dates required for
        the current matrix.

        :param feature_tables: the tables to be used for the current matrix
        :param schema: the name of the features schema
        :type feature_tables: list
        :type schema: str

        :return: none
        :rtype: none
        """
        if matrix_type == 'train':
            indices_query = self.build_labels_query(
                as_of_times = as_of_times,
                final_column = '',
                label_name = label_name,
                label_type = label_type
            )
        elif matrix_type == 'test':
            indices_query = self.get_all_valid_entity_date_combos(
                as_of_times = as_of_times,
                feature_table_names = feature_table_names
            )
        else:
            raise ValueError('Unknown matrix type passed: {}'.format(matrix_type))

        query = """
            CREATE TABLE {features_schema_name}.tmp_entity_date
            AS ({index_query})
        """.format(
            features_schema_name = self.db_config['features_schema_name'],
            index_query = indices_query
        )
        self.engine.execute(query)

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

    def merge_feature_csvs(self, source_filenames, out_filename):
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
        with open(out_filename, 'w') as outfile:
            writer = csv.writer(outfile)
            source_filehandles = [open(fname) for fname in source_filenames]
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
                        logging.info('Found headers: %s', headers)
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
                for fh in source_filehandles:
                    fh.close()
