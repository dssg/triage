# -*- coding: utf-8 -*-
from itertools import chain
import sqlalchemy.sql.expression as ex
import logging

from .sql import make_sql_clause, to_sql_name, CreateTableAs, InsertFromSelect
from triage.component.architect.utils import remove_schema_from_table_name
from triage.database_reflection import table_exists
from triage.component.architect.feature_block import FeatureBlock
from .from_obj import FromObj
from descriptors import cachedproperty

from .imputations import (
    ImputeMean,
    ImputeConstant,
    ImputeZero,
    ImputeZeroNoFlag,
    ImputeNullCategory,
    ImputeBinaryMode,
    ImputeError,
    IMPUTATION_COLNAME_SUFFIX
)

available_imputations = {
    "mean": ImputeMean,
    "constant": ImputeConstant,
    "zero": ImputeZero,
    "zero_noflag": ImputeZeroNoFlag,
    "null_category": ImputeNullCategory,
    "binary_mode": ImputeBinaryMode,
    "error": ImputeError,
}

AGGFUNCS_NEED_MULTIPLE_VALUES = set(['stddev', 'stddev_samp', 'variance', 'var_samp'])


class SpacetimeAggregation(FeatureBlock):
    def __init__(
        self,
        aggregates,
        groups,
        from_obj,
        intervals=None,
        entity_column=None,
        prefix=None,
        suffix=None,
        date_column=None,
        output_date_column=None,
        drop_interim_tables=True,
        *args,
        **kwargs
    ):
        """
        Args:
            aggregates: collection of Aggregate objects.
            from_obj: defines the from clause, e.g. the name of the table. can use
            groups: a list of expressions to group by in the aggregation or a dictionary
                pairs group: expr pairs where group is the alias (used in column names)
            entity_column: the group level found in the state table (e.g., "entity_id")
            prefix: prefix for aggregation tables and column names, defaults to from_obj
            suffix: suffix for aggregation table, defaults to "aggregation"
            intervals: the intervals to aggregate over. either a list of
                datetime intervals, e.g. ["1 month", "1 year"], or
                a dictionary of group : intervals pairs where
                group is a group in groups and intervals is a collection
                of datetime intervals, e.g. {"address_id": ["1 month", "1 year]}
            entity_column: the group level found in the cohort table (e.g., "entity_id")
            date_column: name of date column in from_obj, defaults to "date"
            output_date_column: name of date column in aggregated output, defaults to "date"
        """
        super().__init__(*args, **kwargs)
        self.groups = (
            groups if isinstance(groups, dict) else {str(g): g for g in groups}
        )
        if isinstance(intervals, dict):
            self.intervals = intervals
        elif intervals:
            self.intervals = {g: intervals for g in self.groups}
        else:
            self.intervals = {g: ["all"] for g in self.groups}

        self.date_column = date_column if date_column else "date"
        self.output_date_column = output_date_column if output_date_column else "date"
        self.aggregates = aggregates
        self.from_obj = make_sql_clause(from_obj, ex.text)
        self.entity_column = entity_column if entity_column else "entity_id"
        self.prefix = prefix if prefix else self.features_table_name_without_schema
        self.drop_interim_tables = drop_interim_tables

    def get_table_name(self, group=None, imputed=False):
        """
        Returns name for table for the given group
        """
        if imputed:
            return self.final_feature_table_name
        prefix = self.features_table_name_without_schema
        if group is None:
            name = '"%s_%s"' % (prefix, "aggregation")
        else:
            name = '"%s"' % to_sql_name("%s_%s" % (prefix, group))
        schema = '"%s".' % to_sql_name(self.features_schema_name) if self.features_schema_name else ""
        return "%s%s" % (schema, name)

    def get_drops(self):
        """
        Generate drop queries for this aggregation

        Returns: a dictionary of group : drop pairs where
            group are the same keys as groups
            drop is a raw drop table query for the corresponding table
        """
        return [
            "DROP TABLE IF EXISTS %s;" % self.get_table_name(group)
            for group in self.groups
        ]

    def get_drop(self, imputed=False):
        """
        Generate a drop table statement for the aggregation table
        Returns: string sql query
        """
        return "DROP TABLE IF EXISTS %s" % self.get_table_name(imputed=imputed)

    def get_create_schema(self):
        """
        Generate a create schema statement
        """
        if self.features_schema_name is not None:
            return "CREATE SCHEMA IF NOT EXISTS %s" % self.features_schema_name

    def imputed_flag_column_names(self):
        # format the query that gets column names,
        # excluding indices from result
        feature_names_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}' AND
                  table_schema = '{schema}' AND
                  column_name like '%%{suffix}'
        """.format(
            table=remove_schema_from_table_name(self.get_table_name(imputed=True)),
            schema=self.features_schema_name or 'public',
            suffix=IMPUTATION_COLNAME_SUFFIX
        )
        feature_names = [
            row[0]
            for row in self.db_engine.execute(feature_names_query)
        ]
        return feature_names

    def _basecol_of_impflag(self, impflag_col):
        # we don't want to add redundant imputation flags. for a given source
        # column and time interval, all of the functions will have identical 
        # sets of rows that needed imputation
        # to reliably merge these, we lookup the original aggregate that produced
        # the function, and see its available functions. we expect exactly one of
        # these functions to end the column name and remove it if so
        if hasattr(self.colname_aggregate_lookup[impflag_col], 'functions'):
            agg_functions = self.colname_aggregate_lookup[impflag_col].functions
            used_function = next(funcname for funcname in agg_functions if impflag_col.endswith(funcname))
            if used_function in AGGFUNCS_NEED_MULTIPLE_VALUES:
                return impflag_col
            else:
                return impflag_col.rstrip('_' + used_function)
        else:
            logging.warning("Imputation flag merging is not implemented for "
                            "AggregateExpression objects that don't define an aggregate "
                            "function (e.g. composites)")
            return impflag_col

    def _get_impute_select(self, impute_cols, nonimpute_cols, partitionby=None):

        imprules = self.get_imputation_rules()
        used_impflags = set()

        # check if we're missing any columns relative to the full set and raise an
        # exception if we are
        missing_cols = set(imprules.keys()) - set(nonimpute_cols + impute_cols)
        if len(missing_cols) > 0:
            raise ValueError("Missing columns in get_impute_create: %s" % missing_cols)

        # key columns and date column
        query = ""

        # pre-sort and iterate through the combined set to ensure column order
        for col in sorted(nonimpute_cols + impute_cols):
            # just pass through columns that don't require imputation (no nulls found)
            if col in nonimpute_cols:
                query += '\n,"%s"' % col

            # for columns that do require imputation, include SQL to do the imputation work
            # and a flag for whether the value was imputed
            if col in impute_cols:
                impute_rule = imprules[col]

                try:
                    imputer = available_imputations[impute_rule["type"]]
                except KeyError as err:
                    raise ValueError(
                        "Invalid imputation type %s for column %s"
                        % (impute_rule.get("type", ""), col)
                    ) from err

                impflag_basecol = self._basecol_of_impflag(col)
                imputer = imputer(column=col, column_base_for_impflag=impflag_basecol, partitionby=partitionby, **impute_rule)

                query += "\n,%s" % imputer.to_sql()
                if not imputer.noflag:
                    # Add an imputation flag for non-categorical columns (this is handeled
                    # for categorical columns with a separate NULL category)
                    # but only add it if another functionally equivalent impflag hasn't already been added
                    impflag_select, impflag_alias = imputer.imputed_flag_select_and_alias()
                    if impflag_alias not in used_impflags:
                        used_impflags.add(impflag_alias)
                        query += "\n,%s as \"%s\" " % (impflag_select, impflag_alias)

        return query

    def get_index(self, imputed=False):
        return "CREATE INDEX ON {} ({})".format(
            self.get_table_name(imputed=imputed),
            self.entity_column,
        )

    def get_creates(self):
        return {
            group: CreateTableAs(self.get_table_name(group), next(iter(sels)).limit(0))
            for group, sels in self.get_selects().items()
        }

    # implement the FeatureBlock interface
    @property
    def feature_columns(self):
        """
        The list of feature columns in the final, postimputation table

        Should exclude any index columns (e.g. entity id, date)
        """
        # start with all columns defined in the feature block.
        # this is important as we don't want to return columns in the final feature table that
        # aren't defined in the feature block (e.g. from an earlier run with more features);
        # this will exclude impflag columns as they are decided after initial features are written
        feature_columns = self.feature_columns_sans_impflags
        impflag_columns = set()

        # our list of imputation flag columns comes from the database,
        # but it may contain columns from prior runs that we didn't specify
        imputation_flag_feature_cols = self.imputed_flag_column_names()
        for feature_column in feature_columns:
            impflag_name = self._basecol_of_impflag(feature_column) + IMPUTATION_COLNAME_SUFFIX
            if impflag_name in imputation_flag_feature_cols:
                impflag_columns.add(impflag_name)
        return feature_columns | impflag_columns

    @property
    def preinsert_queries(self):
        """
        Return all queries that should be run before inserting any data.

        Consists of all queries to drop tables from previous runs, as well as all creates
        needed for this run.

        Returns a list of queries/executable statements
        """
        preinserts = [self.get_drop()] + self.get_drops() + list(self.get_creates().values())
        create_schema = self.get_create_schema()
        if create_schema:
            preinserts.insert(0, create_schema)
        return preinserts

    @property
    def insert_queries(self):
        """
        Return all inserts to populate this data. Each query in this list should be parallelizable.

        Returns a list of queries/executable statements
        """
        return [
            InsertFromSelect(self.get_table_name(group), sel)
            for group, sels in self.get_selects().items()
            for sel in sels
        ]

    @property
    def postinsert_queries(self):
        """
        Return all queries that should be run after inserting all data

        Consists of indexing queries for each group table as well as a
        query to create the aggregation table that encompasses all groups.

        Returns a list of queries/executable statements
        """
        postinserts = [
            "CREATE INDEX ON %s (%s);" % (self.get_table_name(group), groupby)
            for group, groupby in self.groups.items()
        ] + [self.get_create(), self.get_index()]
        if self.drop_interim_tables:
            postinserts += self.get_drops()
        return postinserts

    @property
    def imputation_queries(self):
        """
        Return all queries that should be run to fill in missing data with imputed values.

        Returns a list of queries/executable statements
        """
        if not self.cohort_table_name:
            logging.warning(
                "No cohort table defined in feature_block, cannot create imputation table for %s",
                self.final_feature_table_name
            )
            return []

        if not table_exists(self.cohort_table_name, self.db_engine):
            logging.warning(
                "Cohort table %s does not exist, cannot create imputation table for %s",
                self.cohort_table_name,
                self.final_feature_table_name
            )
            return []

        with self.db_engine.begin() as conn:
            results = conn.execute(self.find_nulls())
            null_counts = results.first().items()
            impute_cols = [col for (col, val) in null_counts if val > 0]
            nonimpute_cols = [col for (col, val) in null_counts if val == 0]
            imp_queries = [
                self.get_drop(imputed=True),  # clear out old imputed data
                self._get_impute_create(impute_cols, nonimpute_cols),  # create the imputed table
                self.get_index(imputed=True),  # index the imputed table
            ]
            if self.drop_interim_tables:
                imp_queries.append(self.get_drop(imputed=False))  # drop the old aggregation table
            return imp_queries

    def preprocess(self):
        create_schema = self.get_create_schema()

        if create_schema is not None:
            with self.db_engine.begin() as conn:
                conn.execute(create_schema)

        if self.materialize_subquery_fromobjs:
            # materialize from obj
            from_obj = FromObj(
                from_obj=self.from_obj.text,
                name=f"{self.features_schema_name}.{self.prefix}",
                knowledge_date_column=self.date_column
            )
            from_obj.maybe_materialize(self.db_engine)
            self.from_obj = from_obj.table

    @cachedproperty
    def colname_aggregate_lookup(self):
        """A reverse lookup from column name to the source collate.Aggregate

        Will error if the Aggregation contains duplicate column names
        """
        lookup = {}
        for group, groupby in self.groups.items():
            intervals = self.intervals[group]
            for interval in intervals:
                for agg in self.aggregates:
                    for col in self._cols_for_aggregate(agg, group, interval, None):
                        if col.name in lookup:
                            raise ValueError("Duplicate feature column name found: ", col.name)
                        lookup[col.name] = agg

        return lookup
            
    def _col_prefix(self, group, interval):
        """
        Helper for creating a column prefix for the group
            group: group clause, for naming columns
            interval: SQL time interval string, or "all"
        Returns: string for a common column prefix for columns in that group and interval
        """
        return "{prefix}_{group}_{interval}_".format(
            prefix=self.prefix, interval=interval, group=group
        )

    def _cols_for_aggregate(self, agg, group, interval, date):
        """
        Helper for getting the sql for a particular aggregate
        Args:
            agg: collate.Aggregate
            interval: SQL time interval string, or "all"
            date: SQL date string
            group: group clause, for naming columns
        Returns: collection of aggregate column SQL strings
        """
        if interval != "all":
            when = "{date_column} >= '{date}'::date - interval '{interval}'".format(
                interval=interval, date=date, date_column=self.date_column
            )
        else:
            when = None
        return agg.get_columns(
            when,
            self._col_prefix(group, interval),
            format_kwargs={"collate_date": date, "collate_interval": interval},
        )

    def _get_aggregates_sql(self, interval, date, group):
        """
        Helper for getting aggregates sql
        Args:
            interval: SQL time interval string, or "all"
            date: SQL date string
            group: group clause, for naming columns
        Returns: collection of aggregate column SQL strings
        """
        return chain(
            *[
                self._cols_for_aggregate(agg, group, interval, date)
                for agg in self.aggregates
            ]
        )

    def index_query(self, imputed=False):
        return "CREATE INDEX ON {} ({}, {})".format(
            self.get_table_name(imputed=imputed),
            self.entity_column,
            self.output_date_column,
        )

    def index_columns(self):
        return sorted(
            [group for group in self.groups.keys()]
            + [self.output_date_column]
        )

    def get_selects(self):
        """
        Constructs select queries for this aggregation

        Returns: a dictionary of group : queries pairs where
            group are the same keys as groups
            queries is a list of Select queries, one for each date in dates
        """
        queries = {}

        for group, groupby in self.groups.items():
            intervals = self.intervals[group]
            queries[group] = []
            for date in self.as_of_dates:
                columns = [
                    make_sql_clause(groupby, ex.text),
                    ex.literal_column("'%s'::date" % date).label(
                        self.output_date_column
                    ),
                ]
                columns += list(
                    chain(
                        *[self._get_aggregates_sql(i, date, group) for i in intervals]
                    )
                )

                gb_clause = make_sql_clause(groupby, ex.literal_column)
                if not self.features_ignore_cohort:
                    from_obj = ex.text(
                        f"(select from_obj.* from ("
                        f"(select * from {self.from_obj}) from_obj join {self.cohort_table_name} cohort on ( "
                        "cohort.entity_id = from_obj.entity_id and "
                        f"cohort.{self.output_date_column} = '{date}'::date)"
                        ")) cohorted_from_obj")
                else:
                    from_obj = self.from_obj
                query = ex.select(columns=columns, from_obj=make_sql_clause(from_obj, ex.text)).group_by(
                    gb_clause
                )
                query = query.where(self.where(date, intervals))

                queries[group].append(query)

        return queries

    def get_imputation_rules(self):
        """
        Constructs a dictionary to lookup an imputation rule from an associated
        column name.

        Returns: a dictionary of column : imputation_rule pairs
        """
        imprules = {}
        for group, groupby in self.groups.items():
            for interval in self.intervals[group]:
                prefix = "{prefix}_{group}_{interval}_".format(
                    prefix=self.prefix, interval=interval, group=group
                )
                for a in self.aggregates:
                    imprules.update(a.column_imputation_lookup(prefix=prefix))
        return imprules

    def where(self, date, intervals):
        """
        Generates a WHERE clause
        Args:
            date: the end date
            intervals: intervals

        Returns: a clause for filtering the from_obj to be between the date and
            the greatest interval
        """
        # upper bound
        w = "{date_column} < '{date}'".format(date_column=self.date_column, date=date)

        # lower bound (if possible)
        if "all" not in intervals:
            greatest = "greatest(%s)" % str.join(
                ",", ["interval '%s'" % i for i in intervals]
            )
            min_date = "'{date}'::date - {greatest}".format(
                date=date, greatest=greatest
            )
            w += "AND {date_column} >= {min_date}".format(
                date_column=self.date_column, min_date=min_date
            )
        if self.feature_start_time is not None:
            w += "AND {date_column} >= '{bot}'::date".format(
                date_column=self.date_column, bot=self.feature_start_time
            )
        return ex.text(w)

    def get_indexes(self):
        """
        Generate create index queries for this aggregation

        Returns: a dictionary of group : index pairs where
            group are the same keys as groups
            index is a raw create index query for the corresponding table
        """
        return {
            group: "CREATE INDEX ON %s (%s, %s);"
            % (self.get_table_name(group), groupby, self.output_date_column)
            for group, groupby in self.groups.items()
        }

    def get_join_table(self):
        """
        Generates a join table, consisting of an entry for each combination of
        groups and dates in the from_obj
        """
        groups = [make_sql_clause(group, ex.text) for group in self.groups.values()]
        intervals = list(set(chain(*self.intervals.values())))

        queries = []
        for date in self.as_of_dates:
            columns = groups + [
                ex.literal_column("'%s'::date" % date).label(self.output_date_column)
            ]
            queries.append(
                ex.select(columns, from_obj=make_sql_clause(self.from_obj, ex.text))
                .where(self.where(date, intervals))
                .group_by(*groups)
            )

        return str.join("\nUNION ALL\n", map(str, queries))

    def get_create(self, join_table=None):
        """
        Generate a single aggregation table creation query by joining
            together the results of get_creates()
        Returns: a CREATE TABLE AS query
        """
        if not join_table:
            join_table = "(%s) t1" % self.get_join_table()
        query = "SELECT * FROM %s\n" % join_table
        for group, groupby in self.groups.items():
            query += " LEFT JOIN %s USING (%s, %s)" % (
                self.get_table_name(group),
                groupby,
                self.output_date_column,
            )

        return "CREATE TABLE %s AS (%s);" % (self.get_table_name(), query)

    def validate(self, conn):
        """
        SpacetimeAggregations ensure that no intervals extend beyond the absolute
        minimum time.
        """
        if self.feature_start_time is not None:
            all_intervals = set(*self.intervals.values())
            for date in self.as_of_dates:
                for interval in all_intervals:
                    if interval == "all":
                        continue
                    # This could be done more efficiently all at once, but doing
                    # it this way allows for nicer error messages.
                    r = conn.execute(
                        "select ('%s'::date - '%s'::interval) < '%s'::date"
                        % (date, interval, self.feature_start_time)
                    )
                    if r.fetchone()[0]:
                        raise ValueError(
                            "date '%s' - '%s' is before feature_start_time ('%s')"
                            % (date, interval, self.feature_start_time)
                        )
                    r.close()
        for date in self.as_of_dates:
            r = conn.execute(
                "select count(*) from %s where %s = '%s'::date"
                % (self.cohort_table_name, self.output_date_column, date)
            )
            if r.fetchone()[0] == 0:
                raise ValueError(
                    "date '%s' is not present in states table ('%s')"
                    % (date, self.cohort_table_name)
                )
            r.close()

    def find_nulls(self, imputed=False):
        """
        Generate query to count number of nulls in each column in the aggregation table

        Returns: a SQL SELECT statement
        """
        query_template = """
            SELECT {cols}
            FROM {state_tbl} t1
            LEFT JOIN {aggs_tbl} t2 USING({group}, {date_col})
            """
        cols_sql = ",\n".join(
            [
                """SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}" """.format(
                    col=column
                )
                for column in self.get_imputation_rules().keys()
            ]
        )

        return query_template.format(
            cols=cols_sql,
            state_tbl=self._cohort_table_sub(),
            aggs_tbl=self.get_table_name(imputed=imputed),
            group=self.entity_column,
            date_col=self.output_date_column,
        )

    def _get_impute_create(self, impute_cols, nonimpute_cols):
        """
        Generates the CREATE TABLE query for the aggregation table with imputation.

        Args:
            impute_cols: a list of column names with null values
            nonimpute_cols: a list of column names without null values

        Returns: a CREATE TABLE AS query
        """

        # key columns and date column
        query = "SELECT %s, %s" % (
            ", ".join(map(str, self.groups.values())),
            self.output_date_column,
        )

        # columns with imputation filling as needed
        query += self._get_impute_select(
            impute_cols, nonimpute_cols, partitionby=self.output_date_column
        )

        # imputation starts from the state table and left joins into the aggregation table
        query += "\nFROM %s t1" % self._cohort_table_sub()
        query += "\nLEFT JOIN %s t2 USING(%s, %s)" % (
            self.get_table_name(),
            self.entity_column,
            self.output_date_column,
        )

        return "CREATE TABLE %s AS (%s)" % (self.get_table_name(imputed=True), query)

    @property
    def feature_columns_sans_impflags(self):
        columns = chain.from_iterable(
            chain.from_iterable(
                self._get_aggregates_sql(interval, "2016-01-01", group)
                for interval in self.intervals[group]
            )
            for (group, groupby) in self.groups.items()
        )
        return {label.name for label in columns}
