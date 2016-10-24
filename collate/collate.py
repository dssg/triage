# -*- coding: utf-8 -*-
from itertools import product, chain


def make_list(a):
        return [a] if not type(a) in (list, tuple) else list(a)


class Select(object):
    """
    Simple SQL select query string builder
    """
    def __init__(self, columns, table, where, groupby):
        """
        Args:
            columns: collection of SQL column strings
            table: source table or SQL query from which to select
            where: SQL where clause
            groupby: SQL group-by clause
        """
        self.columns = columns
        self.table = table
        self.where = where
        self.groupby = groupby

    def get_sql(self):
        columns = str.join(',\n    ', self.columns)
        return ("SELECT {columns}\n"
                "FROM {table}\n"
                "WHERE {where}\n"
                "GROUP BY {groupby}".format(
                    columns=columns, table=self.table,
                    where=self.where, groupby=self.groupby))


class Aggregate(object):
    """
    An object representing one or more SQL aggregate columns in a groupby
    """
    def __init__(self, quantity, function, name=None):
        """
        Args:
            quantity: an SQL string expression for the quantity to aggregate
            function: an SQL aggregate function
            name: a name for the quantity, used in the aggregate column name

        Note that quantity and function can also be collections of the above,
        in which case the cross product of those is used. If quantity is a
        collection than name should also be a collection of the same length.
        """
        self.quantities = make_list(quantity)
        self.functions = make_list(function)

        if name is not None:
            self.quantity_names = make_list(name)
            if len(self.quantity_names) != len(self.quantities):
                raise ValueError("Name length doesn't match quantity length")
        else:
            self.quantity_names = map(str, self.quantities)

    def get_sql(self, when=None, prefix=None):
        """
        Args:
            when: used in a case statement to filter the rows going into the
                aggregation function
            prefix: prefix for column names
        """
        if prefix is None:
            prefix = ""

        name_template = "{prefix}{quantity_name}_{function}"
        if when is None:
            quantity_template = "{function}({quantity})"
        else:
            quantity_template = ("{function}(CASE WHEN {when} "
                                 "THEN {quantity} END)")

        template = "%s AS %s" % (quantity_template, name_template)

        args = product(self.functions, zip(self.quantities,
                                           self.quantity_names))
        return [template.format(function=function, quantity=quantity,
                                when=when, quantity_name=quantity_name,
                                prefix=prefix)
                for function, (quantity, quantity_name) in args]


class SpacetimeAggregation(object):
    def __init__(self, aggregates, intervals, table, groupby, dates,
                 prefix=None, date_column=None):
        """
        Args:
            aggregates: collection of Aggregate objects
            intervals: collection of PostgreSQL time interval strings, or "all"
                e.g. ["1 month", "1 year", "all"]
            table: name of table (or SQL subquery) to select from
            groupby: SQL group by clause
            dates: list of PostgreSQL date strings,
                e.g. ["2012-01-01", "2013-01-01"]
            prefix: name of prefix for column names, defaults to table
            date_column: name of date column in the table, defaults to "date"
        """
        self.aggregates = aggregates
        self.intervals = intervals
        self.table = table
        self.groupby = groupby
        self.dates = dates
        self.prefix = prefix if prefix else table
        self.date_column = date_column if date_column else "date"

    def _get_aggregates_sql(self, interval, date):
        """
        Helper for getting aggregates sql
        Args:
            interval: SQL time interval string, or "all"
            date: SQL date string
        Returns: collection of aggregate column SQL strings
        """
        if interval != 'all':
            when = "'{date}' - {date_column} < interval '{interval}'".format(
                    interval=interval, date=date, date_column=self.date_column)
        else:
            when = None

        prefix = "{prefix}_{groupby}_{interval}_".format(
                prefix=self.prefix, interval=interval.replace(' ', ''),
                groupby=self.groupby)

        return chain(*(a.get_sql(when, prefix) for a in self.aggregates))

    def get_queries(self):
        """
        Constructs select queries for this aggregation

        Returns: one Select object for each date
        """
        queries = []

        for date in self.dates:
            columns = list(chain(*(self._get_aggregates_sql(i, date)
                                   for i in self.intervals)))
            where = "{date_column} < '{date}'".format(
                    date_column=self.date_column, date=date)
            queries.append(Select(columns, self.table, where, self.groupby))

        return queries
