import io
import subprocess

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import pandas as pd
import numpy as np
import polars as pl
import time

from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Matrix
from triage.database_reflection import table_has_data, table_row_count
from triage.tracking import built_matrix, skipped_matrix, errored_matrix
from triage.util.pandas import downcast_matrix
from triage.component.architect.utils import (
    change_datetimes_on_metadata, 
    check_rows_in_files,
    check_entity_ids_in_files,
    remove_entity_id_and_knowledge_dates,
    generate_list_of_files_to_remove,
    generate_gzip,
    remove_unnecessary_files,
)
from triage.component.catwalk.storage import S3Store

class BuilderBase:
    def __init__(
        self,
        db_config,
        matrix_storage_engine,
        engine,
        experiment_hash,
        replace=True,
        include_missing_labels_in_train_as=None,
        run_id=None,
    ):
        self.db_config = db_config
        self.matrix_storage_engine = matrix_storage_engine
        self.db_engine = engine
        self.experiment_hash = experiment_hash
        self.replace = replace
        self.include_missing_labels_in_train_as = include_missing_labels_in_train_as
        self.run_id = run_id
        self.includes_labels = 'labels_table_name' in self.db_config

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
        logger.info(f"Building {len(build_tasks.keys())} matrices")

        for i, (matrix_uuid, task_arguments) in enumerate(build_tasks.items(), start=1):
            logger.info(
                f"Building matrix {matrix_uuid} [{i}/{len(build_tasks.keys())}]"
            )
            self.build_matrix(**task_arguments)
            logger.success(f"Matrix {matrix_uuid} built")

    def _outer_join_query(
        self,
        right_table_name,
        right_column_selections,
        entity_date_table_name,
        additional_conditions="",
        include_index=True,
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
            query = f"""
                SELECT ed.entity_id,
                    ed.as_of_date{"".join(right_column_selections)}
                FROM {entity_date_table_name} ed
                LEFT OUTER JOIN {right_table_name} r
                ON ed.entity_id = r.entity_id AND
                ed.as_of_date = r.as_of_date
                {additional_conditions}
                ORDER BY ed.entity_id,
                        ed.as_of_date
            """
        else:
            query = f"""
                with r as (
                    SELECT ed.entity_id,
                           ed.as_of_date, {"".join(right_column_selections)[2:]}
                    FROM {entity_date_table_name} ed
                    LEFT OUTER JOIN {right_table_name} r
                    ON ed.entity_id = r.entity_id AND
                       ed.as_of_date = r.as_of_date
                       {additional_conditions}
                    ORDER BY ed.entity_id,
                             ed.as_of_date
                ) 
                select {"".join(right_column_selections)[2:] if not column_override else column_override} 
                from r
            """
        
        return query
    

    def _join_no_labels_case(
        self,
        right_column_selections,
        entity_date_table_name,
        include_index=True,
        column_override=None,
    ):
        """
            Generates the label query for predict forward, when there are not labels 
            associated with the matrix. We return 0's as the label. 

            :param right_column_selections: formatted text for the columns to select
            :param entity_date_table_name: name of table containing all valid entity ids and dates

            :type right_column_selections: str
            :type entity_date_table_name: str

            :return: postgresql query for the outer join to the entity-dates table
            :rtype: str
        """
        # put everything into the query
        if include_index:
            query = f"""
                SELECT ed.entity_id,
                    ed.as_of_date{"".join(right_column_selections)}
                FROM {entity_date_table_name} ed
                ORDER BY ed.entity_id,
                        ed.as_of_date
            """
        else:
            query = f"""
                with r as (
                    SELECT ed.entity_id,
                           ed.as_of_date, {"".join(right_column_selections)[2:]}
                    FROM {entity_date_table_name} ed
                    ORDER BY ed.entity_id,
                             ed.as_of_date
                ) 
                select {"".join(right_column_selections)[2:] if not column_override else column_override} 
                from r
            """
        
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
        if matrix_type == "test" or matrix_type == "production" or self.include_missing_labels_in_train_as is not None:
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
            raise ValueError(f"Unknown matrix type passed: {matrix_type}")

        table_name = "_".join([matrix_uuid, "matrix_entity_date"])
        query = f"""
            DROP TABLE IF EXISTS {self.db_config["features_schema_name"]}."{table_name}";
            CREATE TABLE {self.db_config["features_schema_name"]}."{table_name}"
            AS ({indices_query})
        """
        logger.debug(
            f"Creating matrix-specific entity-date table for matrix {matrix_uuid} ",
        )
        logger.debug(f"with query {query}")
        self.db_engine.execute(query)

        return table_name

    def _all_labeled_entity_dates_query(
        self, as_of_time_strings, state, label_name, label_type, label_timespan
    ):
        query = f"""
            SELECT entity_id, as_of_date
            FROM {self.db_config["cohort_table_name"]}
            JOIN {self.db_config["labels_schema_name"]}.{self.db_config["labels_table_name"]} using (entity_id, as_of_date)
            WHERE {state}
            AND as_of_date IN (SELECT (UNNEST (ARRAY{as_of_time_strings}::timestamp[])))
            AND label_name = '{label_name}'
            AND label_type = '{label_type}'
            AND label_timespan = '{label_timespan}'
            AND label is not null
            ORDER BY entity_id, as_of_date
        """
        return query

    def _all_valid_entity_dates_query(self, state, as_of_time_strings):
        query = f"""
            SELECT entity_id, as_of_date
            FROM {self.db_config["cohort_table_name"]}
            WHERE {state}
            AND as_of_date IN (SELECT (UNNEST (ARRAY{as_of_time_strings}::timestamp[])))
            ORDER BY entity_id, as_of_date
        """
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
        logger.spam(f"popped matrix {matrix_uuid} build off the queue")
        if not table_has_data(
            self.db_config["cohort_table_name"], self.db_engine
        ):
            logger.warning("cohort table is not populated, cannot build matrix")
            if self.run_id:
                errored_matrix(self.run_id, self.db_engine)
            return

        if self.includes_labels:
            if not table_has_data(
                    f"{self.db_config['labels_schema_name']}.{self.db_config['labels_table_name']}",
                    self.db_engine,
            ):
                logger.warning("labels table is not populated, cannot build matrix")
                if self.run_id:
                    errored_matrix(self.run_id, self.db_engine)

        matrix_store = self.matrix_storage_engine.get_store(matrix_uuid)
        if not self.replace and matrix_store.exists:
            logger.notice(f"Skipping {matrix_uuid} because matrix already exists")
            if self.run_id:
                skipped_matrix(self.run_id, self.db_engine)
            return

        logger.debug(
            f'Storing matrix {matrix_metadata["matrix_id"]} in {matrix_store.matrix_base_store.path}'
        )
        # make the entity time table and query the labels and features tables
        logger.debug(f"Making entity date table for matrix {matrix_uuid}")
        try:
            entity_date_table_name = self.make_entity_date_table(
                as_of_times,
                label_name,
                label_type,
                matrix_metadata["state"],
                matrix_type,
                matrix_uuid,
                matrix_metadata.get("label_timespan", None),
            )
        except ValueError as e:
            logger.exception(
                "Not able to build entity-date table,  will not build matrix",
            )
            if self.run_id:
                errored_matrix(self.run_id, self.db_engine)
            return
        logger.spam(
            f"Extracting feature group data from database into file for matrix {matrix_uuid}"
        )
        
        feature_queries = self.feature_load_queries(feature_dictionary, entity_date_table_name)
        logger.debug(f"feature queries, number of queries: {len(feature_queries)}")
        
        # when predict forwarding we don't have labels
        logger.info(f"Includes labels?: {self.includes_labels}")
        
        label_query = self.label_load_query(
            label_name,
            label_type,
            entity_date_table_name,
            matrix_metadata["label_timespan"],
        )
        logger.debug(f"Label query while building the matrix: {label_query}")

        output, labels = self.stitch_csvs(feature_queries, label_query, matrix_store, matrix_uuid)
        logger.info(f"matrix stitched, pandas DF returned")
        matrix_store.metadata = matrix_metadata
        matrix_store.matrix_label_tuple = output, labels
        logger.info(f"Saving matrix metadata (yaml) for matrix {matrix_uuid}")
        matrix_store.save_matrix_metadata()

        # If completely archived, save its information to matrices table
        # At this point, existence of matrix already tested, so no need to delete from db
        logging.info(f"Getting all matrix metadata for matrix {matrix_uuid}")
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
            num_observations=row_count[0], #row count is a tuple
            lookback_duration=lookback,
            feature_start_time=matrix_metadata["feature_start_time"],
            feature_dictionary=feature_dictionary,
            matrix_metadata=matrix_metadata,
            built_by_experiment=self.experiment_hash
        )
        logger.info(f"About to save all matrix metadata on DB for matrix {matrix_uuid}")
        # before saving the matrix metadata we need to cast datetimes to str 
        matrix_metadata = change_datetimes_on_metadata(matrix_metadata)
        session = self.sessionmaker()
        session.merge(matrix)
        session.commit()
        session.close()
        if self.run_id:
            built_matrix(self.run_id, self.db_engine)


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

        if self.includes_labels:
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
                #include_index=False,
                include_index=True,
                column_override=label_name
            )
        else: 
            labels_query = self._join_no_labels_case(
                right_column_selections=f", 0 as {label_name}",
                entity_date_table_name=f'"{self.db_config["features_schema_name"]}"."{entity_date_table_name}"',
            )

        return labels_query
    

    def feature_load_queries(self, feature_dictionary, entity_date_table_name):
        """ Loop over tables in features schema, writing the data from each to a csv. Return the full list of feature 
        csv names and the list of all features.
        :param feature_dictionary: a dictionary of feature tables and features to be included in the matrix
        :param entity_date_table_name: the name of the entity date table for the matrix
        :type feature_dictionary: dict
        :type entity_date_table_name: str
        :return: list of csvs containing feature data
        :rtype: list
        """
        # iterate! for each table, make query
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
            ))
        return queries


    def stitch_csvs(self, features_queries, label_query, matrix_store, matrix_uuid):
        """
        Get all features related this matrix_uuid as CSV files, as well as the labels. 
        Join all the csv elements columnwise and create the final matrix. 
        The last column is the label. 

        Args:
            features_queries (list): List of the requried queries to execute to 
                get all the features from this design matrix. 
            label_query (string): The query required to get the label associated 
                to this design matrix. 
            matrix_store (MatrixStorage): Storage path for the project
            matrix_uuid (string): Id of the matrix

        Returns:
            DataFrame: Design downcasted matrix
        """
        logger.debug(f"stitching csvs for matrix {matrix_uuid}")
        connection = self.db_engine.raw_connection()
        cursor = connection.cursor()
        header = "HEADER"

        # starting with features 
        path_ = str(matrix_store.get_storage_directory())
        logger.debug(f"path to store csvs {path_}")

        filenames = []
        for i, query_string in enumerate(features_queries):
            copy_sql = f"COPY ({query_string}) TO STDOUT WITH CSV {header}"
            bio = io.BytesIO()
            cursor.copy_expert(copy_sql, bio)
            bio.seek(0)
            output_ = bio.read()
            
            filenames.append(path_ + "/" + matrix_uuid + "_" + str(i) + ".csv")
            
            matrix_store.save_tmp_csv(output_, path_, matrix_uuid, f"_{str(i)}.csv")

        logger.debug(f"number of feature files to paste for matrix {matrix_uuid}: {len(filenames)}")

        # label
        copy_sql = f"COPY ({label_query}) TO STDOUT WITH CSV {header}"
        bio = io.BytesIO()
        cursor.copy_expert(copy_sql, bio)
        bio.seek(0)
        output_ = bio.read()

        matrix_store.save_tmp_csv(output_, path_, matrix_uuid, "_label.csv")

        # add label file to filenames
        filenames.append(f"{path_}/{matrix_uuid}_label.csv")
        
        # check if the number of rows among all features and label -if any- files are the same
        try: 
            assert check_rows_in_files(filenames, matrix_uuid)
        except AssertionError as e: 
            logger.exception(
                f"Different number of rows among features and label files for matrix uuid {matrix_uuid} ",
            )
            if self.run_id:
                errored_matrix(self.run_id, self.db_engine)
            raise
    
        # check if the entities_id and knowledge_dates are the same among all the features and label files
        try:
            check_entity_ids_in_files(filenames, matrix_uuid)
        except AssertionError as e:
            logger.exception(
                f"Not the same order of entity id and knowledge date in all features and label files for matrix uuid {matrix_uuid}"
            )
            if self.run_id:
                errored_matrix(self.run_id, self.db_engine)
            raise

        # remove first 2 columns on each features and label files -except the first one- 
        verified_filenames = remove_entity_id_and_knowledge_dates(filenames, matrix_uuid)

        # join all files starting with features and ending with label
        files = " ".join(verified_filenames)

        # save joined csvs
        cmd_line = 'paste ' + files + ' -d "," > ' + path_ + "/" + matrix_uuid + ".csv"
        logger.debug(f"About to paste CSVs columnwise for matrix {matrix_uuid} cmd line: {cmd_line}")
        try:
            subprocess.run(cmd_line, shell=True, check=True)
            logger.debug(f"CSV files {files} pasted into {path_}/{matrix_uuid}.csv")
        except subprocess.CalledProcessError as e:      
            logger.error(f"Error pasting CSV files {files}: {e}")
        
        logger.debug(f"about to load csvmatrix with uuid {matrix_uuid} as polars df")
        start = time.time()
        # load as DF with polars
        filename_ = f'{path_}/{matrix_uuid}.csv'
        #df = pd.read_csv(filename_, parse_dates=["as_of_date"])
        df_pl = pl.read_csv(filename_, infer_schema_length=0).with_columns(pl.all().exclude(
            ['entity_id', 'as_of_date']).cast(pl.Float32, strict=False))
        end = time.time()
        logger.debug(f"time to read csv of matrix with uuid {matrix_uuid} (sec): {(end-start)/60}")
        
        # casting entity_id and as_of_date 
        logger.debug(f"casting entity_id and as_of_date")
        start = time.time()
        # define if as_of_date is date or datetime for correct cast
        if len(df_pl.get_column('as_of_date').head(1)[0].split()) > 1: 
            format = "%Y-%m-%d %H:%M:%S"
        else: 
            format = "%Y-%m-%d"

        df_pl = df_pl.with_columns(pl.col("as_of_date").str.to_datetime(format))
        df_pl = df_pl.with_columns(pl.col("entity_id").cast(pl.Int32, strict=False))
        end = time.time()
        logger.debug(f"time casting entity_id and as_of_date of matrix with uuid {matrix_uuid} (sec): {(end-start)/60}")
        
        logger.debug(f"getting labels pandas series from polars data frame")
        
        # getting label series
        labels_pl = df_pl.select(df_pl.columns[-1])
        # convert into pandas series 
        labels_df = labels_pl.to_pandas()
        labels_series = labels_df.squeeze()

        # remove labels from features and return as df
        logger.debug(f"removing labels from main polars df")
        df_pl_aux = df_pl.drop(df_pl.columns[-1])

        # converting from polars to pandas
        logger.debug(f"about to convert polars df into pandas df")
        start = time.time()
        df = df_pl_aux.to_pandas()
        end = time.time()
        logger.debug(f"Time converting from polars to pandas (sec): {(end-start)/60}")
        df.set_index(["entity_id", "as_of_date"], inplace=True)
        logger.debug(f"df data types: {df.dtypes}")
        logger.debug(f"Pandas DF memory usage: {df.memory_usage(deep=True).sum()/1000000} MB")
        # generating gzip file from csv
        generate_gzip(path_, matrix_uuid)

        # if matrix store is S3 
        if isinstance(matrix_store.matrix_base_store, S3Store):
            logger.debug(f"storing {matrix_uuid}.csv.gz on S3")
            matrix_store._save(path_, matrix_store.matrix_base_store.path)

        logger.debug(f"removing csvs files for matrix {matrix_uuid}")
        # addinig _sorted and _fixed files to list of files to rm 
        rm_filenames = generate_list_of_files_to_remove(filenames, matrix_uuid)
        remove_unnecessary_files(rm_filenames, path_, matrix_uuid)

        return df, labels_series

        
        