import json
import logging
import itertools

import contextlib
from sqlalchemy.orm import sessionmaker
from functools import partial
from ohio import PipeTextIO
import yaml

from triage.component.results_schema import Matrix
from triage.database_reflection import table_has_data


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
        except ValueError:
            logging.warning(
                "Not able to build entity-date table due to: %s - will not build matrix",
                exc_info=True,
            )
            return
        logging.info(
            "Extracting feature group data from database into file " "for matrix %s",
            matrix_uuid,
        )
        label_query = self.label_load_query(
            label_name,
            label_type,
            entity_date_table_name,
            matrix_metadata["label_timespan"],
        )

        # stitch together the csvs
        logging.info("Merging feature files for matrix %s", matrix_uuid)
        self.queries_to_matrixstore(
            self.feature_load_queries(feature_dictionary, entity_date_table_name) + [label_query],
            matrix_store
        )
        with matrix_store.metadata_base_store.open('wb') as fd:
            yaml.dump(matrix_metadata, fd, encoding="utf-8")

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
            num_observations=5,
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
        l = []
        for feature_table_name, feature_names in feature_dictionary.items():
            logging.info("Generating feature query for %s", feature_table_name)
            l.append(self._outer_join_query(
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
            ))
        return l

    @property
    def _raw_connections(self):
        while True:
            yield self.db_engine.raw_connection()

    def queries_to_matrixstore(self, queries, matrix_store):
        copy_sqls = (f"COPY ({query}) TO STDOUT WITH CSV HEADER" for query in queries)

        with contextlib.ExitStack() as stack:
            connections = (stack.enter_context(contextlib.closing(conn))
                           for conn in self._raw_connections)
            cursors = (conn.cursor() for conn in connections)

            writers = (partial(cursor.copy_expert, copy_sql)
                       for (cursor, copy_sql) in zip(cursors, copy_sqls))
            pipes = (stack.enter_context(PipeTextIO(writer, buffer_size=100)) for writer in writers)
            row_buffer = (
                itertools.chain(*(
                    line.rstrip('\r\n').split(',')[2:] if i > 0 else line.rstrip('\r\n').split(',')
                    for i, line in enumerate(join)
                ))
                for join in zip(*pipes)
            )
            matrix_store.save(row_buffer)
