import json
import logging

import contextlib
from sqlalchemy.orm import sessionmaker
from functools import partial
from ohio import PipeTextIO
from triage.util.io import IteratorBytesIO

from triage.component.results_schema import Matrix
from triage.database_reflection import table_has_data, table_row_count
from triage.validation_primitives import table_should_have_entity_date_columns, table_should_have_data


class BuilderBase(object):
    def __init__(
        self,
        db_config,
        matrix_storage_engine,
        engine,
        experiment_hash,
        replace=True,
        include_missing_labels_in_train_as=None,
        constrain_memory=False,
    ):
        self.db_config = db_config
        self.matrix_storage_engine = matrix_storage_engine
        self.db_engine = engine
        self.experiment_hash = experiment_hash
        self.replace = replace
        self.include_missing_labels_in_train_as = include_missing_labels_in_train_as
        self.constrain_memory = constrain_memory

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
        include_index=False,
        column_override=None,
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
        
        if include_index:
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
        else:
            query = """
                with r as (
                    SELECT ed.entity_id,
                           ed.as_of_date, {columns}
                    FROM {entity_date_table_name} ed
                    LEFT OUTER JOIN {right_table} r
                    ON ed.entity_id = r.entity_id AND
                       ed.as_of_date = r.as_of_date
                       {more}
                    ORDER BY ed.entity_id,
                             ed.as_of_date
                ) select {columns_maybe_override} from r
            """.format(
                columns="".join(right_column_selections)[2:],
                columns_maybe_override="".join(right_column_selections)[2:] if not column_override else column_override,
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

        # what should the labels table look like?
        # 1. have data
        # 2. entity date/column
        labels_table_name = f"{self.db_config['labels_schema_name']}.{self.db_config['labels_table_name']}"
        if not table_has_data(
            labels_table_name,
            self.db_engine,
        ):
            logging.warning("labels table is not populated, cannot build matrix")
            return

        table_should_have_entity_date_columns(
            labels_table_name,
            self.db_engine
        )

        # what should the feature tables look like?
        # 1. have data
        # 2. entity/date column
        for feature_table in feature_dictionary.keys():
            full_feature_table = \
                f"{self.db_config['features_schema_name']}.{feature_table}"
            table_should_have_data(full_feature_table, self.db_engine)
            table_should_have_entity_date_columns(full_feature_table,  self.db_engine)


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
        except ValueError:
            logging.warning(
                "Not able to build entity-date table due to: %s - will not build matrix",
                exc_info=True,
            )
            return
        feature_queries = self.feature_load_queries(feature_dictionary, entity_date_table_name)
        label_query = self.label_load_query(
            label_name,
            label_type,
            entity_date_table_name,
            matrix_metadata["label_timespan"],
        )

        # stitch together the csvs
        logging.info("Building and saving matrix %s by querying and joining tables", matrix_uuid)
        self._save_matrix(
            queries=feature_queries + [label_query],
            matrix_store=matrix_store,
            matrix_metadata=matrix_metadata
        )

        # If completely archived, save its information to matrices table
        # At this point, existence of matrix already tested, so no need to delete from db
        if matrix_type == "train":
            lookback = matrix_metadata["max_training_history"]
        else:
            lookback = matrix_metadata["test_duration"]

        row_count = table_row_count(
            '{schema}."{table}"'.format(
                schema=self.db_config["features_schema_name"],
                table=entity_date_table_name,
            ),
            self.db_engine
        )

        matrix = Matrix(
            matrix_id=matrix_metadata["matrix_id"],
            matrix_uuid=matrix_uuid,
            matrix_type=matrix_type,
            labeling_window=matrix_metadata["label_timespan"],
            num_observations=row_count,
            lookback_duration=lookback,
            feature_start_time=matrix_metadata["feature_start_time"],
            matrix_metadata=json.dumps(matrix_metadata, sort_keys=True, default=str),
            built_by_experiment=self.experiment_hash
        )
        session = self.sessionmaker()
        session.merge(matrix)
        session.commit()
        session.close()

    def label_load_query(
        self,
        label_name,
        label_type,
        entity_date_table_name,
        label_timespan,
    ):
        """ Query the labels table and write the data to disk in csv format.

        :param as_of_times: the times to be used for the current matrix
        :param label_name: name of the label to be used
        :param label_type: the type of label to be used
        :param entity_date_table_name: the name of the entity date table
        :param label_timespan: the time timespan that labels in matrix will include
        :type label_name: str
        :type label_type: str
        :type entity_date_table_name: str
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
            include_index=False,
            column_override=label_name
        )

        return labels_query

    def feature_load_queries(self, feature_dictionary, entity_date_table_name):
        """ Loop over tables in features schema, writing the data from each to a
        csv. Return the full list of feature csv names and the list of all
        features.

        :param feature_dictionary: a dictionary of feature tables and features
            to be included in the matrix
        :param entity_date_table_name: the name of the entity date table
            for the matrix
        :type feature_dictionary: dict
        :type entity_date_table_name: str

        :return: list of csvs containing feature data
        :rtype: tuple
        """
        # iterate! for each table, make query, write csv, save feature & file names
        queries = []
        for num, (feature_table_name, feature_names) in enumerate(feature_dictionary.items()):
            logging.info("Generating feature query for %s", feature_table_name)
            queries.append(self._outer_join_query(
                right_table_name="{schema}.{table}".format(
                    schema=self.db_config["features_schema_name"],
                    table=feature_table_name,
                ),
                entity_date_table_name='{schema}."{table}"'.format(
                    schema=self.db_config["features_schema_name"],
                    table=entity_date_table_name,
                ),
                right_column_selections=[', "{0}"'.format(fn) for fn in feature_names],
                include_index=True if num==0 else False,
            ))
        return queries

    @property
    def _raw_connections(self):
        while True:
            yield self.db_engine.raw_connection()

    def _save_matrix(self, queries, matrix_store, matrix_metadata):
        """Construct and save a matrix CSV from a list of queries

        The results of each query are expected to return the same number of rows in the same order.
        The columns will be placed alongside each other in the CSV much as a SQL join would.
        However, this code does not deduplicate the columns, so the actual row identifiers
        (e.g. entity id, as of date) should only be present in one of the queries
        unless you want duplicate columns.

        The result, and the given metadata, will be given to the supplied MatrixStore for saving.

        Args:
            queries (iterable) SQL queries
            matrix_store (triage.component.catwalk.storage.CSVMatrixStore)
            matrix_metadata (dict) matrix metadata to save alongside the data
        """
        copy_sqls = (f"COPY ({query}) TO STDOUT WITH CSV HEADER" for query in queries)
        with contextlib.ExitStack() as stack:
            connections = (stack.enter_context(contextlib.closing(conn))
                           for conn in self._raw_connections)
            cursors = (conn.cursor() for conn in connections)

            writers = (partial(cursor.copy_expert, copy_sql)
                       for (cursor, copy_sql) in zip(cursors, copy_sqls))
            pipes = (stack.enter_context(PipeTextIO(writer)) for writer in writers)
            iterable = (
                b','.join(line.rstrip('\r\n').encode('utf-8') for line in join) + b'\n'
                for join in zip(*pipes)
            )
            matrix_store.save(from_fileobj=IteratorBytesIO(iterable), metadata=matrix_metadata)
