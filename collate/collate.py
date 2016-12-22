# -*- coding: utf-8 -*-
from itertools import product, chain
import sqlalchemy.sql.expression as ex

from .sql import make_sql_clause, to_sql_name, CreateTableAs, InsertFromSelect


def make_list(a):
    return [a] if not isinstance(a, list) else a


def make_tuple(a):
    return (a,) if not isinstance(a, tuple) else a


class Aggregate(object):
    """
    An object representing one or more SQL aggregate columns in a groupby
    """
    def __init__(self, quantity, function, order=None):
        """
        Args:
            quantity: SQL for the quantity to aggregate
            function: SQL aggregate function
            order: SQL for order by clause in an ordered set aggregate

        Notes:
            quantity, function, and order can also be lists of the above,
            in which case the cross product of those is used. If quantity is a
            collection than name should also be a collection of the same length.

            quantity can be a tuple of SQL quantities for aggregate functions
            that take multiple arguments, e.g. corr, regr_slope

            quantity can be a dictionary in which case the keys are names
            for the expressions and values are expressions.
        """
        if isinstance(quantity, dict):
            self.quantities = quantity
        else:
            # first convert to list of tuples
            quantities = [make_tuple(q) for q in make_list(quantity)]
            # then dict with name keys
            self.quantities = {to_sql_name(str.join("_", q)): q for q in quantities}

        self.functions = make_list(function)
        self.orders = make_list(order)

    def get_columns(self, when=None, prefix=None, format_kwargs=None):
        """
        Args:
            when: used in a case statement to filter the rows going into the
                aggregation function
            prefix: prefix for column names
            format_kwargs: kwargs to pass to format the aggregate quantity
        Returns:
            collection of SQLAlchemy columns
        """
        if prefix is None:
            prefix = ""
        if format_kwargs is None:
            format_kwargs = {}

        name_template = "{prefix}{quantity_name}_{function}"
        column_template = "{function}({args})"
        arg_template = "{quantity}"
        order_template = ""

        if self.orders != [None]:
            column_template += " WITHIN GROUP (ORDER BY {order_clause})"
            order_template = "CASE WHEN {when} THEN {order} END" if when else "{order}"
        elif when:
            arg_template = "CASE WHEN {when} THEN {quantity} END"

        for function, (quantity_name, quantity), order in product(
                self.functions, self.quantities.items(), self.orders):
            args = str.join(", ", (arg_template.format(when=when, quantity=q)
                                   for q in make_tuple(quantity)))
            order_clause = order_template.format(when=when, order=order)

            kwargs = dict(function=function, args=args, prefix=prefix,
                          order_clause=order_clause,
                          quantity_name=quantity_name, **format_kwargs)

            column = column_template.format(**kwargs).format(**format_kwargs)
            name = name_template.format(**kwargs)

            yield ex.literal_column(column).label(to_sql_name(name))


class Aggregation(object):
    def __init__(self, aggregates, groups, from_obj, prefix=None, suffix=None, schema=None):
        """
        Args:
            aggregates: collection of Aggregate objects.
            from_obj: defines the from clause, e.g. the name of the table. can use 
            groups: a list of expressions to group by in the aggregation or a dictionary
                pairs group: expr pairs where group is the alias (used in column names)
            prefix: prefix for aggregation tables and column names, defaults to from_obj
            suffix: suffix for aggregation table, defaults to "aggregation"
            schema: schema for aggregation tables

        The from_obj and group expressions are passed directly to the
            SQLAlchemy Select object so could be anything supported there.
            For details see:
            http://docs.sqlalchemy.org/en/latest/core/selectable.html

        Aggregates will have {collate_date} in their quantities substituted with the date
        of aggregation.
        """
        self.aggregates = aggregates
        self.from_obj = make_sql_clause(from_obj, ex.text)
        self.groups = groups if isinstance(groups, dict) else {str(g): g for g in groups}
        self.prefix = prefix if prefix else str(from_obj)
        self.suffix = suffix if suffix else "aggregation"
        self.schema = schema

    def _get_aggregates_sql(self, group):
        """
        Helper for getting aggregates sql
        Args:
            group: group clause, for naming columns
        Returns: collection of aggregate column SQL strings
        """
        prefix = "{prefix}_{group}_".format(
                prefix=self.prefix, group=group)

        return chain(*(a.get_columns(prefix=prefix)
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
            columns = [groupby]
            columns += self._get_aggregates_sql(group)

            gb_clause = make_sql_clause(groupby, ex.literal_column)
            query = ex.select(columns=columns, from_obj=self.from_obj)\
                      .group_by(gb_clause)

            queries[group] = [query]

        return queries

    def get_table_name(self, group=None):
        """
        Returns name for table for the given group
        """
        if group is None:
            name = '"%s_%s"' % (self.prefix, self.suffix)
        else:
            name = '"%s"' % to_sql_name("%s_%s" % (self.prefix, group))
        schema = '"%s".' % self.schema if self.schema else ''
        return "%s%s" % (schema, name)

    def get_creates(self):
        """
        Construct create queries for this aggregation
        Args:
            selects: the dictionary of select queries to use
                if None, use self.get_selects()
                this allows you to customize select queries before creation

        Returns:
            a dictionary of group : create pairs where
                group are the same keys as groups
                create is a CreateTableAs object
        """
        return {group: CreateTableAs(self.get_table_name(group),
                                     next(iter(sels)).limit(0))
                for group, sels in self.get_selects().items()}

    def get_inserts(self):
        """
        Construct insert queries from this aggregation
        Args:
            selects: the dictionary of select queries to use
                if None, use self.get_selects()
                this allows you to customize select queries before creation

        Returns:
            a dictionary of group : inserts pairs where
                group are the same keys as groups
                inserts is a list of InsertFromSelect objects
        """
        return {group: [InsertFromSelect(self.get_table_name(group), sel) for sel in sels]
                for group, sels in self.get_selects().items()}

    def get_drops(self):
        """
        Generate drop queries for this aggregation

        Returns: a dictionary of group : drop pairs where
            group are the same keys as groups
            drop is a raw drop table query for the corresponding table
        """
        return {group: "DROP TABLE IF EXISTS %s;" % self.get_table_name(group)
                for group in self.groups}

    def get_indexes(self):
        """
        Generate create index queries for this aggregation

        Returns: a dictionary of group : index pairs where
            group are the same keys as groups
            index is a raw create index query for the corresponding table
        """
        return {group: "CREATE INDEX ON %s (%s);" %
                (self.get_table_name(group), groupby)
                for group, groupby in self.groups.items()}

    def get_join_table(self):
        """
        Generate a query for a join table
        """
        return ex.Select(columns=self.groups.values(), from_obj=self.from_obj)\
                 .group_by(*self.groups.values())

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
            query += "LEFT JOIN %s USING (%s)" % (
                    self.get_table_name(group), groupby)

        return "CREATE TABLE %s AS (%s);" % (self.get_table_name(), query)

    def get_drop(self):
        """
        Generate a drop table statement for the aggregation table
        Returns: string sql query
        """
        return "DROP TABLE IF EXISTS %s" % self.get_table_name()

    def get_create_schema(self):
        """
        Generate a create schema statement
        """
        if self.schema is not None:
            return "CREATE SCHEMA IF NOT EXISTS %s" % self.schema

    def execute(self, conn):
        """
        Execute all SQL statements to create final aggregation table.
        Args:
            conn: the SQLAlchemy connection on which to execute
        """
        creates = self.get_creates()
        drops = self.get_drops()
        indexes = self.get_indexes()
        inserts = self.get_inserts()

        trans = conn.begin()
        if self.schema is not None:
            conn.execute(self.get_create_schema())

        for group in self.groups:
            conn.execute(drops[group])
            conn.execute(creates[group])
            for insert in inserts[group]:
                conn.execute(insert)
            conn.execute(indexes[group])

        conn.execute(self.get_drop())
        conn.execute(self.get_create())
        trans.commit()


class SpacetimeAggregation(Aggregation):
    def __init__(self, aggregates, groups, intervals, from_obj, dates,
                 prefix=None, suffix=None, schema=None, date_column=None, output_date_column=None):
        """
        Args:
            aggregates: collection of Aggregate objects
            from_obj: defines the from clause, e.g. the name of the table
            groups: a list of expressions to group by in the aggregation or a dictionary
                pairs group: expr pairs where group is the alias (used in column names)
            intervals: the intervals to aggregate over. either a list of
                datetime intervals, e.g. ["1 month", "1 year"], or
                a dictionary of group : intervals pairs where
                group is a group in groups and intervals is a collection
                of datetime intervals, e.g. {"address_id": ["1 month", "1 year]}
            dates: list of PostgreSQL date strings,
                e.g. ["2012-01-01", "2013-01-01"]
            prefix: prefix for column names, defaults to from_obj
            suffix: suffix for aggregation table, defaults to "aggregation"
            date_column: name of date column in from_obj, defaults to "date"
            output_date_column: name of date column in aggregated output, defaults to "date"

        The from_obj and group arguments are passed directly to the
            SQLAlchemy Select object so could be anything supported there.
            For details see:
            http://docs.sqlalchemy.org/en/latest/core/selectable.html
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

                # upper bound on date_column by date
                where = ex.text("{date_column} < '{date}'".format(
                        date_column=self.date_column, date=date))

                gb_clause = make_sql_clause(groupby, ex.literal_column)
                query = ex.select(columns=columns, from_obj=self.from_obj)\
                          .where(where)\
                          .group_by(gb_clause)

                if 'all' not in intervals:
                    greatest = "greatest(%s)" % str.join(
                            ",", ["interval '%s'" % i for i in intervals])
                    query = query.where(ex.text(
                        "{date_column} >= '{date}'::date - {greatest}".format(
                            date_column=self.date_column, date=date,
                            greatest=greatest)))

                queries[group].append(query)

        return queries

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

    def get_create(self, join_table=None):
        """
        Generate a single aggregation table creation query by joining
            together the results of get_creates()
        Returns: a CREATE TABLE AS query
        """
        if not join_table:
            join_table = '(%s) t1' % self.get_join_table()

        query = ("SELECT * FROM %s\n"
                 "CROSS JOIN (select unnest('{%s}'::date[]) as %s) t2\n") % (
                join_table, str.join(',', self.dates), self.output_date_column)
        for group, groupby in self.groups.items():
            query += "LEFT JOIN %s USING (%s, %s)" % (
                    self.get_table_name(group), groupby, self.output_date_column)

        return "CREATE TABLE %s AS (%s);" % (self.get_table_name(), query)
