import io
import json
import logging
import pandas

from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Matrix
from triage.database_reflection import table_has_data
from triage.util.pandas import downcast_matrix


class BuilderBase(object):
    def __init__(
        self,
        db_config,
        matrix_storage_engine,
        engine,
        experiment_hash,
        replace=True,
        include_missing_labels_in_train_as=None,
    ):
        self.db_config = db_config
        self.matrix_storage_engine = matrix_storage_engine
        self.db_engine = engine
        self.experiment_hash = experiment_hash
        self.replace = replace
        self.include_missing_labels_in_train_as = include_missing_labels_in_train_as

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    def validate(self):
        for expected_db_config_val in [
            "features_schema_name",
            "cohort_table_name",
            "labels_schema_name",
            "labels_table_name",
        ]:
            if expected_db_config_val not in self.db_config:
                raise ValueError(
                    "{} needed in db_config".format(expected_db_config_val)
                )

    def build_all_matrices(self, build_tasks):
        logging.info("Building %s matrices", len(build_tasks.keys()))

        for i, (matrix_uuid, task_arguments) in enumerate(build_tasks.items()):
            logging.info(
                f"Building matrix {matrix_uuid} ({i}/{len(build_tasks.keys())})"
            )
            self.build_matrix(**task_arguments)
            logging.debug(f"Matrix {matrix_uuid} built")

    def _outer_join_query(
        self,
        right_table_name,
        right_column_selections,
        entity_date_table_name,
        additional_conditions="",
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
            columns="".join(right_column_selections),
            feature_schema=self.db_config["features_schema_name"],
            entity_date_table_name=entity_date_table_name,
            right_table=right_table_name,
            more=additional_conditions,
        )
        return query

    def make_entity_date_table(
        self,
        as_of_times,
        label_name,
        label_type,
        state,
        matrix_type,
        matrix_uuid,
        label_timespan,
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
        if matrix_type == "test" or self.include_missing_labels_in_train_as is not None:
            indices_query = self._all_valid_entity_dates_query(
                as_of_time_strings=as_of_time_strings, state=state
            )
        elif matrix_type == "train":
            indices_query = self._all_labeled_entity_dates_query(
                as_of_time_strings=as_of_time_strings,
                state=state,
                label_name=label_name,
                label_type=label_type,
                label_timespan=label_timespan,
            )
        else:
            raise ValueError("Unknown matrix type passed: {}".format(matrix_type))

        table_name = "_".join([matrix_uuid, "matrix_entity_date"])
        query = """
            DROP TABLE IF EXISTS {features_schema_name}."{table_name}";
            CREATE TABLE {features_schema_name}."{table_name}"
            AS ({index_query})
        """.format(
            features_schema_name=self.db_config["features_schema_name"],
            table_name=table_name,
            index_query=indices_query,
        )
        logging.debug(
            "Creating matrix-specific entity-date table for matrix " "%s with query %s",
            matrix_uuid,
            query,
        )
        self.db_engine.execute(query)

        return table_name

    def _all_labeled_entity_dates_query(
        self, as_of_time_strings, state, label_name, label_type, label_timespan
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
            states_table=self.db_config["cohort_table_name"],
            state_string=state,
            labels_schema_name=self.db_config["labels_schema_name"],
            labels_table_name=self.db_config["labels_table_name"],
            l_name=label_name,
            l_type=label_type,
            timespan=label_timespan,
            times=as_of_time_strings,
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
            states_table=self.db_config["cohort_table_name"],
            state_string=state,
            times=as_of_time_strings,
        )
        if not table_has_data(
            self.db_config["cohort_table_name"], self.db_engine
        ):
            raise ValueError("Required cohort table does not exist")
        return query


class MatrixBuilder(BuilderBase):
    def build_matrix(
        self,
        as_of_times,
        label_name,
        label_type,
        feature_dictionary,
        matrix_metadata,
        matrix_uuid,
        matrix_type,
    ):
        """ Write a design matrix to disk with the specified paramters.

        :param as_of_times: datetimes to be included in the matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param feature_dictionary: a dictionary of feature tables and features
                                   to be included in the matrix
        :param matrix_metadata: a dictionary of metadata about the matrix
        :param matrix_uuid: a unique id for the matrix
        :param matrix_type: the type (train/test) of matrix
        :type as_of_times: list
        :type label_name: str
        :type label_type: str
        :type feature_dictionary: dict
        :type matrix_metadata: dict
        :type matrix_uuid: str
        :type matrix_type: str

        :return: none
        :rtype: none
        """
        logging.info("popped matrix %s build off the queue", matrix_uuid)
        if not table_has_data(
            self.db_config["cohort_table_name"], self.db_engine
        ):
            logging.warning("cohort table is not populated, cannot build matrix")
            return
        if not table_has_data(
            "{}.{}".format(
                self.db_config["labels_schema_name"],
                self.db_config["labels_table_name"],
            ),
            self.db_engine,
        ):
            logging.warning("labels table is not populated, cannot build matrix")
            return

        matrix_store = self.matrix_storage_engine.get_store(matrix_uuid)
        if not self.replace and matrix_store.exists:
            logging.info("Skipping %s because matrix already exists", matrix_uuid)
            return

        logging.info(
            "Creating matrix %s > %s",
            matrix_metadata["matrix_id"],
            matrix_store.matrix_base_store.path,
        )
        # make the entity time table and query the labels and features tables
        logging.info("Making entity date table for matrix %s", matrix_uuid)
        try:
            entity_date_table_name = self.make_entity_date_table(
                as_of_times,
                label_name,
                label_type,
                matrix_metadata["state"],
                matrix_type,
                matrix_uuid,
                matrix_metadata["label_timespan"],
            )
        except ValueError as e:
            logging.warning(
                "Not able to build entity-date table due to: %s - will not build matrix",
                exc_info=True,
            )
            return
        logging.info(
            "Extracting feature group data from database into file " "for matrix %s",
            matrix_uuid,
        )
        dataframes = self.load_features_data(
            as_of_times, feature_dictionary, entity_date_table_name, matrix_uuid
        )
        logging.info(f"Feature data extracted for matrix {matrix_uuid}")
        logging.info(
            "Extracting label data from database into file for " "matrix %s",
            matrix_uuid,
        )
        labels_df = self.load_labels_data(
            label_name,
            label_type,
            entity_date_table_name,
            matrix_uuid,
            matrix_metadata["label_timespan"],
        )
        dataframes.insert(0, labels_df)

        logging.info(f"Label data extracted for matrix {matrix_uuid}")
        # stitch together the csvs
        logging.info("Merging feature files for matrix %s", matrix_uuid)
        output = self.merge_feature_csvs(dataframes, matrix_uuid)
        logging.info(f"Features data merged for matrix {matrix_uuid}")

        matrix_store.metadata = matrix_metadata
        # store the matrix
        labels = output.pop(matrix_store.label_column_name)
        matrix_store.matrix_label_tuple = output, labels
        matrix_store.save()
        logging.info("Matrix {matrix_uuid} saved")
        # If completely archived, save its information to matrices table
        # At this point, existence of matrix already tested, so no need to delete from db
        if matrix_type == "train":
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
            matrix_metadata=json.dumps(matrix_metadata, sort_keys=True, default=str),
            built_by_experiment=self.experiment_hash
        )
        session = self.sessionmaker()
        session.merge(matrix)
        session.commit()
        session.close()

    def load_labels_data(
        self,
        label_name,
        label_type,
        entity_date_table_name,
        matrix_uuid,
        label_timespan,
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
        if self.include_missing_labels_in_train_as is None:
            label_predicate = "r.label"
        elif self.include_missing_labels_in_train_as is False:
            label_predicate = "coalesce(r.label, 0)"
        elif self.include_missing_labels_in_train_as is True:
            label_predicate = "coalesce(r.label, 1)"
        else:
            raise ValueError(
                'incorrect value "{}" for include_missing_labels_in_train_as'.format(
                    self.include_missing_labels_in_train_as
                )
            )

        labels_query = self._outer_join_query(
            right_table_name="{schema}.{table}".format(
                schema=self.db_config["labels_schema_name"],
                table=self.db_config["labels_table_name"],
            ),
            entity_date_table_name='"{schema}"."{table}"'.format(
                schema=self.db_config["features_schema_name"],
                table=entity_date_table_name,
            ),
            right_column_selections=", {} as {}".format(label_predicate, label_name),
            additional_conditions="""AND
                r.label_name = '{name}' AND
                r.label_type = '{type}' AND
                r.label_timespan = '{timespan}'
            """.format(
                name=label_name, type=label_type, timespan=label_timespan
            ),
        )

        return self.query_to_df(labels_query)

    def load_features_data(
        self, as_of_times, feature_dictionary, entity_date_table_name, matrix_uuid
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
        feature_dfs = []
        for feature_table_name, feature_names in feature_dictionary.items():
            logging.info("Retrieving feature data from %s", feature_table_name)
            features_query = self._outer_join_query(
                right_table_name="{schema}.{table}".format(
                    schema=self.db_config["features_schema_name"],
                    table=feature_table_name,
                ),
                entity_date_table_name='{schema}."{table}"'.format(
                    schema=self.db_config["features_schema_name"],
                    table=entity_date_table_name,
                ),
                # collate imputation shouldn't leave any nulls and we double-check
                # the imputed table in FeatureGenerator.create_all_tables() but as
                # a final check, raise a divide by zero error on export if the
                # database encounters any during the outer join
                right_column_selections=[', "{0}"'.format(fn) for fn in feature_names],
            )
            feature_dfs.append(self.query_to_df(features_query))

        return feature_dfs

    def query_to_df(self, query_string, header="HEADER"):
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
        logging.debug("Copying to CSV query %s", query_string)
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query_string, head=header
        )
        conn = self.db_engine.raw_connection()
        cur = conn.cursor()
        out = io.StringIO()
        cur.copy_expert(copy_sql, out)
        out.seek(0)
        df = pandas.read_csv(out, parse_dates=["as_of_date"])
        df.set_index(["entity_id", "as_of_date"], inplace=True)
        return downcast_matrix(df)

    def merge_feature_csvs(self, dataframes, matrix_uuid):
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

        for i, df in enumerate(dataframes):
            if df.index.names != ["entity_id", "as_of_date"]:
                raise ValueError(
                    f"index must be entity_id and as_of_date, value was {df.index}"
                )
            # check for any nulls. the labels, understood to be the first file,
            # can have nulls but no features should. therefore, skip the first dataframe
            if i > 0:
                columns_with_nulls = [
                    column for column in df.columns if df[column].isnull().values.any()
                ]
                if len(columns_with_nulls) > 0:
                    raise ValueError(
                        "Imputation failed for the following features: %s"
                        % columns_with_nulls
                    )
            i += 1

        big_df = dataframes[1].join(dataframes[2:] + [dataframes[0]])
        return big_df
