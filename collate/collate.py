# -*- coding: utf-8 -*-
from itertools import product, chain
from sqlalchemy.sql.expression import select, literal_column


def make_list(a):
        return [a] if not type(a) in (list, tuple) else list(a)


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
            self.quantity_names = [x.replace('"', '') for x in self.quantities]

    def get_columns(self, when=None, prefix=None):
        """
        Args:
            when: used in a case statement to filter the rows going into the
                aggregation function
            prefix: prefix for column names
        Returns:
            collection of SQLAlchemy columns
        """
        if prefix is None:
            prefix = ""

        name_template = "{prefix}{quantity_name}_{function}"
        if when is None:
            column_template = "{function}({quantity})"
        else:
            column_template = ("{function}(CASE WHEN {when} "
                               "THEN {quantity} END)")

        format_kwargs = dict(prefix=prefix, when=when)

        for function, (quantity, quantity_name) in product(
                self.functions, zip(self.quantities, self.quantity_names)):
            format_kwargs.update(quantity=quantity, function=function,
                                 quantity_name=quantity_name)
            column = column_template.format(**format_kwargs)
            name = name_template.format(**format_kwargs)

            yield literal_column(column).label(name)


class SpacetimeAggregation(object):
    def __init__(self, aggregates, intervals, from_obj, group_by, dates,
                 prefix=None, date_column=None):
        """
        Args:
            aggregates: collection of Aggregate objects
            intervals: collection of PostgreSQL time interval strings, or "all"
                e.g. ["1 month', "1 year", "all"]
            from_obj: defines the from clause, e.g. the name of the table
            group_by: defines the groupby, e.g. the name of a column
            dates: list of PostgreSQL date strings,
                e.g. ["2012-01-01", "2013-01-01"]
            prefix: name of prefix for column names, defaults to from_obj
            date_column: name of date column in from_obj, defaults to "date"

        The from_obj and group_by arguments are passed directly to the
            SQLAlchemy Select object so could be anything supported there.
            For details see:
            http://docs.sqlalchemy.org/en/latest/core/selectable.html
        """
        self.aggregates = aggregates
        self.intervals = intervals
        self.from_obj = from_obj
        self.group_by = group_by
        self.dates = dates
        self.prefix = prefix if prefix else str(from_obj)
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
            when = "'{date}' < {date_column} + interval '{interval}'".format(
                    interval=interval, date=date, date_column=self.date_column)
        else:
            when = None

        prefix = "{prefix}_{group_by}_{interval}_".format(
                prefix=self.prefix, interval=interval.replace(' ', ''),
                group_by=self.group_by)

        return chain(*(a.get_columns(when, prefix) for a in self.aggregates))

    def get_queries(self):
        """
        Constructs select queries for this aggregation

        Returns: one SQLAlchemy Select query object per date
        """
        queries = []

        for date in self.dates:
            columns = list(chain(*(self._get_aggregates_sql(i, date)
                                   for i in self.intervals)))
            where = "{date_column} < '{date}'".format(
                    date_column=self.date_column, date=date)
            queries.append(select(columns=columns, from_obj=self.from_obj)
                                 .where(where)
                                 .group_by(self.group_by))

        return queries
