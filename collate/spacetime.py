# -*- coding: utf-8 -*-
from itertools import chain
import sqlalchemy.sql.expression as ex

from .sql import make_sql_clause
from .collate import Aggregation


class SpacetimeAggregation(Aggregation):
    def __init__(self, aggregates, groups, intervals, from_obj, dates,
                 prefix=None, suffix=None, schema=None, date_column=None, output_date_column=None):
        """
        Args:
            intervals: the intervals to aggregate over. either a list of
                datetime intervals, e.g. ["1 month", "1 year"], or
                a dictionary of group : intervals pairs where
                group is a group in groups and intervals is a collection
                of datetime intervals, e.g. {"address_id": ["1 month", "1 year]}
            dates: list of PostgreSQL date strings,
                e.g. ["2012-01-01", "2013-01-01"]
            date_column: name of date column in from_obj, defaults to "date"
            output_date_column: name of date column in aggregated output, defaults to "date"

        For all other arguments see collate.Aggregation
        """
        Aggregation.__init__(self,
                             aggregates=aggregates,
                             from_obj=from_obj,
                             groups=groups,
                             prefix=prefix,
                             suffix=suffix,
                             schema=schema)

        if isinstance(intervals, dict):
            self.intervals = intervals
        else:
            self.intervals = {g: intervals for g in self.groups}
        self.dates = dates
        self.date_column = date_column if date_column else "date"
        self.output_date_column = output_date_column if output_date_column else "date"

    def _get_aggregates_sql(self, interval, date, group):
        """
        Helper for getting aggregates sql
        Args:
            interval: SQL time interval string, or "all"
            date: SQL date string
            group: group clause, for naming columns
        Returns: collection of aggregate column SQL strings
        """
        if interval != 'all':
            when = "{date_column} >= '{date}'::date - interval '{interval}'".format(
                    interval=interval, date=date, date_column=self.date_column)
        else:
            when = None

        prefix = "{prefix}_{group}_{interval}_".format(
                prefix=self.prefix, interval=interval,
                group=group)

        return chain(*(a.get_columns(when, prefix, format_kwargs={"collate_date": date})
                       for a in self.aggregates))

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
                columns = [groupby,
                           ex.literal_column("'%s'::date"
                                             % date).label(self.output_date_column)]
                columns += list(chain(*(self._get_aggregates_sql(
                        i, date, group) for i in intervals)))

                gb_clause = make_sql_clause(groupby, ex.literal_column)
                query = ex.select(columns=columns, from_obj=self.from_obj)\
                          .group_by(gb_clause)
                query = query.where(self.where(date, intervals))

                queries[group].append(query)

        return queries

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
        w = "{date_column} < '{date}'".format(
                            date_column=self.date_column, date=date)

        # lower bound
        if 'all' not in intervals:
            greatest = "greatest(%s)" % str.join(
                    ",", ["interval '%s'" % i for i in intervals])
            w += "AND {date_column} >= '{date}'::date - {greatest}".format(
                    date_column=self.date_column, date=date,
                    greatest=greatest)

        return ex.text(w)

    def get_indexes(self):
        """
        Generate create index queries for this aggregation

        Returns: a dictionary of group : index pairs where
            group are the same keys as groups
            index is a raw create index query for the corresponding table
        """
        return {group: "CREATE INDEX ON %s (%s, %s);" %
                (self.get_table_name(group), groupby, self.output_date_column)
                for group, groupby in self.groups.items()}

    def get_join_table(self):
        """
        Generates a join table, consisting of an entry for each combination of
        groups and dates in the from_obj
        """
        groups = list(self.groups.values())
        intervals = list(set(chain(*self.intervals.values())))

        queries = []
        for date in self.dates:
            columns = groups + [ex.literal_column("'%s'::date" % date).label(
                    self.output_date_column)]
            queries.append(ex.select(columns, from_obj=self.from_obj)
                             .where(self.where(date, intervals))
                             .group_by(*groups))

        return str.join("\nUNION ALL\n", map(str, queries))

    def get_create(self, join_table=None):
        """
        Generate a single aggregation table creation query by joining
            together the results of get_creates()
        Returns: a CREATE TABLE AS query
        """
        if not join_table:
            join_table = '(%s) t1' % self.get_join_table()
        query = "SELECT * FROM %s\n" % join_table
        for group, groupby in self.groups.items():
            query += " LEFT JOIN %s USING (%s, %s)" % (
                    self.get_table_name(group), groupby, self.output_date_column)

        return "CREATE TABLE %s AS (%s);" % (self.get_table_name(), query)
