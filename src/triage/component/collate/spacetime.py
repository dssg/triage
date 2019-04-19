# -*- coding: utf-8 -*-
from itertools import chain
import sqlalchemy.sql.expression as ex
from descriptors import cachedproperty

from .sql import make_sql_clause
from .collate import Aggregation


class SpacetimeAggregation(Aggregation):
    def __init__(
        self,
        aggregates,
        groups,
        intervals,
        from_obj,
        dates,
        state_table,
        state_group=None,
        prefix=None,
        suffix=None,
        schema=None,
        date_column=None,
        output_date_column=None,
        input_min_date=None,
        join_with_cohort_table=False,
    ):
        """
        Args:
            intervals: the intervals to aggregate over. either a list of
                datetime intervals, e.g. ["1 month", "1 year"], or
                a dictionary of group : intervals pairs where
                group is a group in groups and intervals is a collection
                of datetime intervals, e.g. {"address_id": ["1 month", "1 year]}
            dates: list of PostgreSQL date strings,
                e.g. ["2012-01-01", "2013-01-01"]
            state_table: schema.table to query for valid state_group/date combinations
            state_group: the group level found in the state table (e.g., "entity_id")
            date_column: name of date column in from_obj, defaults to "date"
            output_date_column: name of date column in aggregated output, defaults to "date"
            input_min_date: minimum date for which rows shall be included, defaults
                to no absolute time restrictions on the minimum date of included rows

        For all other arguments see collate.Aggregation
        """
        Aggregation.__init__(
            self,
            aggregates=aggregates,
            from_obj=from_obj,
            groups=groups,
            state_table=state_table,
            state_group=state_group,
            prefix=prefix,
            suffix=suffix,
            schema=schema,
        )

        if isinstance(intervals, dict):
            self.intervals = intervals
        else:
            self.intervals = {g: intervals for g in self.groups}
        self.dates = dates
        self.date_column = date_column if date_column else "date"
        self.output_date_column = output_date_column if output_date_column else "date"
        self.input_min_date = input_min_date
        self.join_with_cohort_table = join_with_cohort_table

    def _state_table_sub(self):
        """Helper function to ensure we only include state table records
        in our set of input dates and after the input_min_date.
        """
        datestr = ", ".join(["'%s'::date" % dt for dt in self.dates])
        mindtstr = (
            " AND %s >= '%s'::date" % (self.output_date_column, self.input_min_date)
            if self.input_min_date is not None
            else ""
        )
        return """(
        SELECT *
        FROM {st}
        WHERE {datecol} IN ({datestr})
        {mindtstr})""".format(
            st=self.state_table,
            datecol=self.output_date_column,
            datestr=datestr,
            mindtstr=mindtstr,
        )

    @cachedproperty
    def colname_aggregate_lookup(self):
        """A reverse lookup from column name to the source collate.Aggregate

        Will error if the Aggregation contains duplicate column names
        """
        lookup = {}
        for group, groupby in self.groups.items():
            intervals = self.intervals[group]
            for interval in intervals:
                date = self.dates[0]
                for agg in self.aggregates:
                    for col in self._cols_for_aggregate(agg, group, interval, date):
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
            for date in self.dates:
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
                if self.join_with_cohort_table:
                    from_obj = ex.text(
                        f"(select from_obj.* from ("
                        f"(select * from {self.from_obj}) from_obj join {self.state_table} cohort on ( "
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
        if self.input_min_date is not None:
            w += "AND {date_column} >= '{bot}'::date".format(
                date_column=self.date_column, bot=self.input_min_date
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
        for date in self.dates:
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
        if self.input_min_date is not None:
            all_intervals = set(*self.intervals.values())
            for date in self.dates:
                for interval in all_intervals:
                    if interval == "all":
                        continue
                    # This could be done more efficiently all at once, but doing
                    # it this way allows for nicer error messages.
                    r = conn.execute(
                        "select ('%s'::date - '%s'::interval) < '%s'::date"
                        % (date, interval, self.input_min_date)
                    )
                    if r.fetchone()[0]:
                        raise ValueError(
                            "date '%s' - '%s' is before input_min_date ('%s')"
                            % (date, interval, self.input_min_date)
                        )
                    r.close()
        for date in self.dates:
            r = conn.execute(
                "select count(*) from %s where %s = '%s'::date"
                % (self.state_table, self.output_date_column, date)
            )
            if r.fetchone()[0] == 0:
                raise ValueError(
                    "date '%s' is not present in states table ('%s')"
                    % (date, self.state_table)
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
            state_tbl=self._state_table_sub(),
            aggs_tbl=self.get_table_name(imputed=imputed),
            group=self.state_group,
            date_col=self.output_date_column,
        )

    def get_impute_create(self, impute_cols, nonimpute_cols):
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
        query += "\nFROM %s t1" % self._state_table_sub()
        query += "\nLEFT JOIN %s t2 USING(%s, %s)" % (
            self.get_table_name(),
            self.state_group,
            self.output_date_column,
        )

        return "CREATE TABLE %s AS (%s)" % (self.get_table_name(imputed=True), query)
