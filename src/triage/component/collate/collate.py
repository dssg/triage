# -*- coding: utf-8 -*-
from numbers import Number
from itertools import product
import sqlalchemy.sql.expression as ex
import re
import logging
from triage.database_reflection import table_exists
from triage.component.architect.utils import remove_schema_from_table_name

from .sql import make_sql_clause, to_sql_name, CreateTableAs, InsertFromSelect


def make_list(a):
    return [a] if not isinstance(a, list) else a


def make_tuple(a):
    return (a,) if not isinstance(a, tuple) else a


DISTINCT_REGEX = re.compile(r"distinct[ (]")


def split_distinct(quantity):
    # Only support distinct clauses with one-argument quantities
    if len(quantity) != 1:
        return ("", quantity)
    q = quantity[0]
    if DISTINCT_REGEX.match(q):
        return "distinct ", (q[8:].lstrip(" "),)
    else:
        return "", (q,)


class AggregateExpression(object):
    def __init__(
        self,
        aggregate1,
        aggregate2,
        operator,
        cast=None,
        operator_str=None,
        expression_template=None,
    ):
        """
        Args:
            aggregate1: first aggregate
            aggregate2: second aggregate
            operator: string of SQL operator, e.g. "+"
            cast: optional string to put after aggregate1, e.g. "*1.0", "::decimal"
            operator_str: optional name of operator to use, defaults to operator
            expression_template: optional formatting template with the following keywords:
                name1, operator, name2
        """
        self.aggregate1 = aggregate1
        self.aggregate2 = aggregate2
        self.operator = operator
        self.cast = cast if cast else ""
        self.operator_str = operator if operator_str else operator
        self.expression_template = (
            expression_template if expression_template else "{name1}{operator}{name2}"
        )

    def column_imputation_lookup(self, prefix=None):
        lookup1 = self.aggregate1.column_imputation_lookup()
        lookup2 = self.aggregate2.column_imputation_lookup()

        return dict(
            (
                prefix
                + self.expression_template.format(
                    name1=c1, operator=self.operator_str, name2=c2
                ),
                lookup1[c1],
            )
            for c1, c2 in product(lookup1.keys(), lookup2.keys())
        )

    def alias(self, expression_template):
        """
        Set the expression template used for naming columns of an AggregateExpression
        Returns: self, for chaining
        """
        self.expression_template = expression_template
        return self

    def get_columns(self, when=None, prefix=None, format_kwargs=None):
        if prefix is None:
            prefix = ""

        columns1 = self.aggregate1.get_columns(when)
        columns2 = self.aggregate2.get_columns(when)

        for c1, c2 in product(columns1, columns2):
            c = ex.literal_column(
                "({}{} {} {})".format(c1, self.cast, self.operator, c2)
            )
            yield c.label(
                prefix
                + self.expression_template.format(
                    name1=c1.name, operator=self.operator_str, name2=c2.name
                )
            )

    def __add__(self, other):
        return AggregateExpression(self, other, "+")

    def __sub__(self, other):
        return AggregateExpression(self, other, "-")

    def __mul__(self, other):
        return AggregateExpression(self, other, "*")

    def __div__(self, other):
        return AggregateExpression(self, other, "/", "*1.0")

    def __truediv__(self, other):
        return AggregateExpression(self, other, "/", "*1.0")

    def __lt__(self, other):
        return AggregateExpression(self, other, "<")

    def __le__(self, other):
        return AggregateExpression(self, other, "<=")

    def __eq__(self, other):
        return AggregateExpression(self, other, "=")

    def __ne__(self, other):
        return AggregateExpression(self, other, "!=")

    def __gt__(self, other):
        return AggregateExpression(self, other, ">")

    def __ge__(self, other):
        return AggregateExpression(self, other, ">=")

    def __or__(self, other):
        return AggregateExpression(self, other, "or", operator_str="|")

    def __and__(self, other):
        return AggregateExpression(self, other, "and", operator_str="&")


class Aggregate(AggregateExpression):
    """
    An object representing one or more SQL aggregate columns in a groupby
    """

    def __init__(self, quantity, function, impute_rules, order=None, coltype=None):
        """
        Args:
            quantity: SQL for the quantity to aggregate
            function: SQL aggregate function
            impute_rules: dictionary of rules mapping functions to imputation methods
            order: SQL for order by clause in an ordered set aggregate
            coltype: SQL type for the column in the generated features table

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
            # make quantity values tuples
            self.quantities = {k: make_tuple(q) for k, q in quantity.items()}
        else:
            # first convert to list of tuples
            quantities = [make_tuple(q) for q in make_list(quantity)]
            # then dict with name keys
            self.quantities = {to_sql_name(str.join("_", q)): q for q in quantities}

        self.functions = make_list(function)
        self.orders = make_list(order)
        self.impute_rules = impute_rules
        self.coltype = coltype

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
        coltype_template = ""
        column_template = "{function}({distinct}{args}){order_clause}{filter}{coltype_cast}"
        arg_template = "{quantity}"
        order_template = ""
        filter_template = ""

        if self.orders != [None]:
            order_template += " WITHIN GROUP (ORDER BY {order})"
        if when:
            filter_template = " FILTER (WHERE {when})"

        if self.coltype is not None:
            coltype_template = "::{coltype}"

        for function, (quantity_name, quantity), order in product(
            self.functions, self.quantities.items(), self.orders
        ):
            distinct, quantity = split_distinct(quantity)
            args = str.join(", ", (arg_template.format(quantity=q) for q in quantity))
            order_clause = order_template.format(order=order)
            filter = filter_template.format(when=when)
            coltype_cast = coltype_template.format(coltype=self.coltype)

            if order is not None:
                if len(quantity_name) > 0:
                    quantity_name += "_"
                quantity_name += to_sql_name(order)

            kwargs = dict(
                function=function,
                args=args,
                prefix=prefix,
                distinct=distinct,
                order_clause=order_clause,
                quantity_name=quantity_name,
                filter=filter,
                coltype_cast=coltype_cast,
                **format_kwargs
            )

            column = column_template.format(**kwargs).format(**format_kwargs)
            name = name_template.format(**kwargs)

            yield ex.literal_column(column).label(to_sql_name(name))

    def column_imputation_lookup(self, prefix=None):
        """
        Args:
            prefix: prefix for column names
        Returns:
            dictionary mapping columns to appropriate imputation rule
        """
        if prefix is None:
            prefix = ""

        name_template = "{prefix}{quantity_name}_{function}"

        lkup = {}
        for function, (quantity_name, quantity), order in product(
            self.functions, self.quantities.items(), self.orders
        ):

            if order is not None:
                if len(quantity_name) > 0:
                    quantity_name += "_"
                quantity_name += to_sql_name(order)

            kwargs = dict(function=function, prefix=prefix, quantity_name=quantity_name)

            name = name_template.format(**kwargs)

            # requires an imputation rule defined for any function
            # type used by the aggregate (or catch-all with 'all')
            try:
                lkup[name] = dict(
                    self.impute_rules.get(function) or self.impute_rules["all"],
                    coltype=self.impute_rules["coltype"],
                )
            except KeyError as err:
                raise ValueError(
                    "Must provide an imputation rule for every aggregation "
                    + "function (or 'all'). No rule found for %s" % name
                ) from err

        return lkup


def maybequote(elt, quote_override=None):
    "Quote for passing to SQL if necessary, based upon the python type"

    def quote_string(string):
        return "'{}'".format(string)

    if quote_override is None:
        if isinstance(elt, Number):
            return elt
        else:
            return quote_string(elt)
    elif quote_override:
        return quote_string(elt)
    else:
        return elt


class Compare(Aggregate):
    """
    A simple shorthand to automatically create many comparisons against one column
    """

    def __init__(
        self,
        col,
        op,
        choices,
        function,
        impute_rules,
        order=None,
        coltype=None,
        include_null=False,
        maxlen=None,
        op_in_name=True,
        quote_choices=None,
    ):
        """
        Args:
            col: the column name (or equivalent SQL expression)
            op: the SQL operation (e.g., '=' or '~' or 'LIKE')
            choices: A list or dictionary of values. When a dictionary is
                passed, the keys are a short name for the value.
            function: (from Aggregate)
            impute_rules: (from Aggregate)
            order: (from Aggregate)
            include_null: Add an extra `{col} is NULL` if True (default False).
                 May also be non-boolean, in which case its truthiness determines
                 the behavior and the value is used as the value short name.
            maxlen: The maximum length of aggregate quantity names, if specified.
                Names longer than this will be truncated.
            op_in_name: Include the operator in aggregate names (default False)
            quote_choices: Override smart quoting if present (default None)

        A simple helper method to easily create many comparison columns from
        one source column by comparing it against many values. It effectively
        creates many quantities of the form "({col} {op} {elt})::INT" for elt
        in choices. It automatically quotes strings appropriately and leaves
        numbers unquoted. The type of the comparison is converted to an
        integer so it can easily be used with 'sum' (for total count) and
        'avg' (for relative fraction) aggregate functions.

        By default, the aggregates are named "{col}_{op}_{elt}", but the
        operator may be ommitted if `op_in_name=False`. This name can become
        long and exceed the maximum column name length. If ``maxlen`` is
        specified then any aggregate name longer than ``maxlen`` gets
        truncated with a number appended to ensure that they remain unique and
        identifiable (but note that sequntial ordering is not preserved).
        """
        if type(choices) is not dict:
            choices = {k: k for k in choices}
        opname = "_{}_".format(op) if op_in_name else "_"
        d = {
            "{}{}{}".format(col, opname, nickname): "({} {} {})::INT".format(
                col, op, maybequote(choice, quote_choices)
            )
            for nickname, choice in choices.items()
        }
        if include_null is True:
            include_null = "_NULL"
        if include_null:
            d["{}_{}".format(col, include_null)] = "({} is NULL)::INT".format(col)
        if maxlen is not None and any(len(k) > maxlen for k in d.keys()):
            for i, k in enumerate(list(d.keys())):
                d["%s_%02d" % (k[: maxlen - 3], i)] = d.pop(k)

        Aggregate.__init__(self, d, function, impute_rules, order, coltype)


class Categorical(Compare):
    """
    A simple shorthand to automatically create many equality comparisons against one column
    """

    def __init__(
        self,
        col,
        choices,
        function,
        impute_rules,
        order=None,
        op_in_name=False,
        coltype=None,
        **kwargs
    ):
        """
        Create a Compare object with an equality operator, ommitting the `=`
        from the generated aggregation names. See Compare for more details.

        As a special extension, Compare's 'include_null' keyword option may be
        enabled by including the value `None` in the choices list. Multiple
        None values are ignored.
        """
        if None in choices:
            kwargs["include_null"] = True
            choices.remove(None)
        elif type(choices) is dict and None in choices.values():
            ks = [k for k, v in choices.items() if v is None]
            for k in ks:
                choices.pop(k)
                kwargs["include_null"] = str(k)
        Compare.__init__(
            self,
            col,
            "=",
            choices,
            function,
            impute_rules,
            order,
            coltype,
            op_in_name=op_in_name,
            **kwargs
        )
