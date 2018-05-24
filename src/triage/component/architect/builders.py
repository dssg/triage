import io
import json
import logging
import pandas
import os

import s3fs
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse


from triage.component import metta
from triage.component.results_schema import Matrix


class BuilderBase(object):
    def __init__(
        self,
        db_config,
        matrix_directory,
        engine,
        replace=True,
        include_missing_labels_in_train_as=None,
    ):
        self.db_config = db_config
        self.matrix_directory = matrix_directory
        self.db_engine = engine
        self.replace = replace
        self.include_missing_labels_in_train_as = include_missing_labels_in_train_as

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    def validate(self):
        for expected_db_config_val in [
            'features_schema_name',
            'sparse_state_table_name',
            'labels_schema_name',
            'labels_table_name'
        ]:
            if expected_db_config_val not in self.db_config:
                raise ValueError('{} needed in db_config'.format(expected_db_config_val))

    def build_all_matrices(self, build_tasks):
        logging.info('Building %s matrices', len(build_tasks.keys()))

        for i, (matrix_uuid, task_arguments) in enumerate(build_tasks.items()):
            logging.info(f"Building matrix {matrix_uuid} ({i}/{len(build_tasks.keys())})")
            self.build_matrix(**task_arguments)
            logging.debug(f"Matrix {matrix_uuid} built")

    def _outer_join_query(
        self,
        right_table_name,
        right_column_selections,
        entity_date_table_name,
        additional_conditions=''
    ):
        """ Given a (features or labels) table, a list of times, columns to
        select, and (optionally) a set of join conditions, perform an outer
        join to the entity date table.

        :param right_table_name: the name of the right (feature/label) table
        :param right_column_selections: formatted text for the columns to select
        :param entity_date_table_name: name of table containing all valid entity ids and dates
        :param additional_conditions: formatted text for additional join
                                      conditions
        :type right_table_name: str
        :type right_column_selections: str
        :type entity_date_table_name: str
        :type additional_conditions: str

        :return: postgresql query for the outer join to the entity-dates table
        :rtype: str
        """

        # put everything into the query
        query = """
            SELECT ed.entity_id,
                   ed.as_of_date{columns}
            FROM {entity_date_table_name} ed
            LEFT OUTER JOIN {right_table} r
            ON ed.entity_id = r.entity_id AND
               ed.as_of_date = r.as_of_date
               {more}
            ORDER BY ed.entity_id,
                     ed.as_of_date
        """.format(
            columns=''.join(right_column_selections),
            feature_schema=self.db_config['features_schema_name'],
            entity_date_table_name=entity_date_table_name,
            right_table=right_table_name,
            more=additional_conditions
        )
        return(query)

    def make_entity_date_table(
        self,
        as_of_times,
        label_name,
        label_type,
        state,
        matrix_type,
        matrix_uuid,
        label_timespan
    ):
        """ Make a table containing the entity_ids and as_of_dates required for
        the current matrix.

        :param as_of_times: the times to be used for the current matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param state: the entity state to be used in the matrix
        :param matrix_type: the type (train/test) of matrix
        :param matrix_uuid: a unique id for the matrix
        :param label_timespan: the time timespan that labels in matrix will include
        :type as_of_times: list
        :type label_name: str
        :type label_type: str
        :type state: str
        :type matrix_type: str
        :type matrix_uuid: str
        :type label_timespan: str

        :return: table name
        :rtype: str
        """

        as_of_time_strings = [str(as_of_time) for as_of_time in as_of_times]
        if matrix_type == 'test' or self.include_missing_labels_in_train_as is not None:
            indices_query = self._all_valid_entity_dates_query(
                as_of_time_strings=as_of_time_strings,
                state=state
            )
        elif matrix_type == 'train':
            indices_query = self._all_labeled_entity_dates_query(
                as_of_time_strings=as_of_time_strings,
                state=state,
                label_name=label_name,
                label_type=label_type,
                label_timespan=label_timespan
            )
        else:
            raise ValueError('Unknown matrix type passed: {}'.format(matrix_type))

        table_name = '_'.join([matrix_uuid, 'matrix_entity_date'])
        query = """
            DROP TABLE IF EXISTS {features_schema_name}."{table_name}";
            CREATE TABLE {features_schema_name}."{table_name}"
            AS ({index_query})
        """.format(
            features_schema_name=self.db_config['features_schema_name'],
            table_name=table_name,
            index_query=indices_query
        )
        logging.info('Creating matrix-specific entity-date table for matrix '
                     '%s with query %s', matrix_uuid, query)
        self.db_engine.execute(query)

        return table_name

    def _all_labeled_entity_dates_query(
        self,
        as_of_time_strings,
        state,
        label_name,
        label_type,
        label_timespan
    ):
        query = """
            SELECT entity_id, as_of_date
            FROM {states_table}
            JOIN {labels_schema_name}.{labels_table_name} using (entity_id, as_of_date)
            WHERE {state_string}
            AND as_of_date IN (SELECT (UNNEST (ARRAY{times}::timestamp[])))
            AND label_name = '{l_name}'
            AND label_type = '{l_type}'
            AND label_timespan = '{timespan}'
            AND label is not null
            ORDER BY entity_id, as_of_date
        """.format(
            states_table=self.db_config['sparse_state_table_name'],
            state_string=state,
            labels_schema_name=self.db_config['labels_schema_name'],
            labels_table_name=self.db_config['labels_table_name'],
            l_name=label_name,
            l_type=label_type,
            timespan=label_timespan,
            times=as_of_time_strings
        )

        return query

    def _all_valid_entity_dates_query(self, state, as_of_time_strings):
        query = """
            SELECT entity_id, as_of_date
            FROM {states_table}
            WHERE {state_string}
            AND as_of_date IN (SELECT (UNNEST (ARRAY{times}::timestamp[])))
            ORDER BY entity_id, as_of_date
        """.format(
            states_table=self.db_config['sparse_state_table_name'],
            state_string=state,
            times=as_of_time_strings
        )
        return query


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
        logging.info('popped matrix %s build off the queue', matrix_uuid)
        matrix_filename = os.path.join(
            matrix_directory,
            '{}.csv'.format(matrix_uuid)
        )

        # The output directory is local or in s3
        path_parsed = urlparse(matrix_filename)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if scheme in ('', 'file'):
            if not self.replace and os.path.exists(matrix_filename):
                logging.info('Skipping %s because matrix already exists', matrix_filename)
                return
        elif scheme == 's3':
            if not self.replace and s3fs.S3FileSystem().exists(matrix_filename):
                logging.info('Skipping %s because matrix already exists', matrix_filename)
                return
        else:
            raise ValueError(f"""URL scheme not supported:
              {scheme} (from {matrix_filename})
            """)

        logging.info('Creating matrix %s > %s', matrix_metadata['matrix_id'], matrix_filename)
        # make the entity time table and query the labels and features tables
        logging.info('Making entity date table for matrix %s', matrix_uuid)
        entity_date_table_name = self.make_entity_date_table(
            as_of_times,
            label_name,
            label_type,
            matrix_metadata['state'],
            matrix_type,
            matrix_uuid,
            matrix_metadata['label_timespan']
        )
        logging.info('Extracting feature group data from database into file '
                     'for matrix %s', matrix_uuid)
        features_csv_names = self.write_features_data(
            as_of_times,
            feature_dictionary,
            entity_date_table_name,
            matrix_uuid
        )
        logging.info(f"Feature data extracted for matrix {matrix_uuid}")
        try:
            logging.info('Extracting label data from database into file for '
                         'matrix %s', matrix_uuid)
            labels_csv_name = self.write_labels_data(
                label_name,
                label_type,
                entity_date_table_name,
                matrix_uuid,
                matrix_metadata['label_timespan']
            )
            features_csv_names.insert(0, labels_csv_name)

            logging.info(f"Label data extracted for matrix {matrix_uuid}")
            # stitch together the csvs
            logging.info('Merging feature files for matrix %s', matrix_uuid)
            output = self.merge_feature_csvs(
                features_csv_names,
                matrix_directory,
                matrix_uuid
            )
            logging.info(f"Features data merged for matrix {matrix_uuid}")
        finally:
            # clean up files and database before finishing
            for csv_name in features_csv_names:
                self.remove_file(csv_name)
        try:
            # store the matrix
            logging.info('Archiving matrix %s with metta', matrix_uuid)
            metta.archive_matrix(
                matrix_config=matrix_metadata,
                df_matrix=output,
                overwrite=True,
                directory=self.matrix_directory,
                format='csv'
            )
            logging.info("Matrix {matrix_uuid} archived (using metta)")
            # If completely archived, save its information to matrices table
            # At this point, existence of matrix already tested, so no need to delete from db
            if matrix_type == 'train':
                lookback = matrix_metadata["max_training_history"]
            else:
                lookback = matrix_metadata["test_duration"]

            matrix = Matrix(
                matrix_id=matrix_metadata["matrix_id"],
                matrix_uuid=matrix_uuid,
                matrix_type=matrix_type,
                labeling_window=matrix_metadata["label_timespan"],
                num_observations=len(output),
                lookback_duration=lookback,
                feature_start_time=matrix_metadata["feature_start_time"],
                matrix_metadata=json.dumps(matrix_metadata, sort_keys=True, default=str)
            )
            session = self.sessionmaker()
            session.add(matrix)
            session.commit()
            session.close()

        finally:
            if isinstance(output, str):
                os.remove(output)


    def write_labels_data(
        self,
        label_name,
        label_type,
        entity_date_table_name,
        matrix_uuid,
        label_timespan
    ):
        """ Query the labels table and write the data to disk in csv format.

        :param as_of_times: the times to be used for the current matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param entity_date_table_name: the name of the entity date table
        :param matrix_uuid: a unique id for the matrix
        :param label_timespan: the time timespan that labels in matrix will include
        :type label_name: str
        :type label_type: str
        :type entity_date_table_name: str
        :type matrix_uuid: str
        :type label_timespan: str

        :return: name of csv containing labels
        :rtype: str
        """
        csv_name = os.path.join(
            self.matrix_directory,
            '{}-{}.csv'.format(matrix_uuid, self.db_config['labels_table_name'])
        )
        if self.include_missing_labels_in_train_as is None:
            label_predicate = 'r.label'
        elif self.include_missing_labels_in_train_as is False:
            label_predicate = 'coalesce(r.label, 0)'
        elif self.include_missing_labels_in_train_as is True:
            label_predicate = 'coalesce(r.label, 1)'
        else:
            raise ValueError(
                'incorrect value "{}" for include_missing_labels_in_train_as'.format(
                    self.include_missing_labels_in_train_as
                )
            )

        labels_query = self._outer_join_query(
            right_table_name='{schema}.{table}'.format(
                schema=self.db_config['labels_schema_name'],
                table=self.db_config['labels_table_name']
            ),
            entity_date_table_name='"{schema}"."{table}"'.format(
                schema=self.db_config['features_schema_name'],
                table=entity_date_table_name
            ),
            right_column_selections=', {} as {}'.format(label_predicate, label_name),
            additional_conditions='''AND
                r.label_name = '{name}' AND
                r.label_type = '{type}' AND
                r.label_timespan = '{timespan}'
            '''.format(
                name=label_name,
                type=label_type,
                timespan=label_timespan
            )
        )

        self.write_to_csv(labels_query, csv_name)
        return(csv_name)

    def write_features_data(
        self,
        as_of_times,
        feature_dictionary,
        entity_date_table_name,
        matrix_uuid
    ):
        """ Loop over tables in features schema, writing the data from each to a
        csv. Return the full list of feature csv names and the list of all
        features.

        :param as_of_times: the times to be included in the matrix
        :param feature_dictionary: a dictionary of feature tables and features
            to be included in the matrix
        :param entity_date_table_name: the name of the entity date table
            for the matrix
        :param matrix_uuid: a human-readable id for the matrix
        :type as_of_times: list
        :type feature_dictionary: dict
        :type entity_date_table_name: str
        :type matrix_uuid: str

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
            features_query = self._outer_join_query(
                right_table_name='{schema}.{table}'.format(
                    schema=self.db_config['features_schema_name'],
                    table=feature_table_name
                ),
                entity_date_table_name='{schema}."{table}"'.format(
                    schema=self.db_config['features_schema_name'],
                    table=entity_date_table_name
                ),
                # collate imputation shouldn't leave any nulls and we double-check
                # the imputed table in FeatureGenerator.create_all_tables() but as
                # a final check, raise a divide by zero error on export if the
                # database encounters any during the outer join
                right_column_selections=[', "{0}"'.format(fn) for fn in feature_names]
            )
            self.write_to_csv(features_query, csv_name)
            features_csv_names.append(csv_name)

        return(features_csv_names)

    def write_to_csv(self, query_string, file_name, header='HEADER'):
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
        logging.debug('Copying to CSV query %s', query_string)
        try:
            copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
                query=query_string,
                head=header
            )
            conn = self.db_engine.raw_connection()
            cur = conn.cursor()
            cur.copy_expert(copy_sql, matrix_csv)
        finally:
            self.close_filehandle(file_name)


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
        for i, filehandle in enumerate(source_filehandles):
            df = pandas.read_csv(filehandle)
            df.set_index(['entity_id', 'as_of_date'], inplace=True)
            dataframes.append(df)

            # check for any nulls. the labels, understood to be the first file,
            # can have nulls but no features should. therefore, skip the first dataframe
            if i > 0:
                columns_with_nulls = [
                    column
                    for column in df.columns
                    if df[column].isnull().values.any()
                ]
                if len(columns_with_nulls) > 0:
                    raise ValueError(
                        "Imputation failed for the following features: %s" %
                        columns_with_nulls
                    )
            i += 1

        big_df = dataframes[1].join(dataframes[2:] + [dataframes[0]])
        return big_df
