import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from numbers import Number
from itertools import product, chain
import sqlalchemy.sql.expression as ex
import re
from descriptors import cachedproperty

from .sql import make_sql_clause, to_sql_name, CreateTableAs, InsertFromSelect
from .imputations import (
    ImputeMean,
    ImputeConstant,
    ImputeZero,
    ImputeZeroNoFlag,
    ImputeNullCategory,
    ImputeBinaryMode,
    ImputeError,
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


class NoAggregateFunctionError(ValueError):
    pass


def make_list(a):
    return [a] if not isinstance(a, list) else a


def make_tuple(a):
    return (a,) if not isinstance(a, tuple) else a


DISTINCT_REGEX = re.compile(r"distinct[ (]")
AGGFUNCS_NEED_MULTIPLE_VALUES = set(['stddev', 'stddev_samp', 'variance', 'var_samp'])


def split_distinct(quantity):
    # Only support distinct clauses with one-argument quantities
    if len(quantity) != 1:
        return ("", quantity)
    q = quantity[0]
    if DISTINCT_REGEX.match(q):
        return "distinct ", (q[8:].lstrip(" "),)
    else:
        return "", (q,)


class AggregateExpression:
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


class Aggregation:
    def __init__(
        self,
        aggregates,
        groups,
        from_obj,
        state_table,
        state_group=None,
        prefix=None,
        suffix=None,
        schema=None,
    ):
        """
        Args:
            aggregates: collection of Aggregate objects.
            from_obj: defines the from clause, e.g. the name of the table. can use
            groups: a list of expressions to group by in the aggregation or a dictionary
                pairs group: expr pairs where group is the alias (used in column names)
            state_table: schema.table to query for comprehensive set of state_group entities
                regardless of what exists in the from_obj
            state_group: the group level found in the state table (e.g., "entity_id")
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
        self.groups = (
            groups if isinstance(groups, dict) else {str(g): g for g in groups}
        )
        self.state_table = state_table
        self.state_group = state_group if state_group else "entity_id"
        self.prefix = prefix if prefix else str(from_obj)
        self.suffix = suffix if suffix else "aggregation"
        self.schema = schema

    @cachedproperty
    def colname_aggregate_lookup(self):
        """A reverse lookup from column name to the source collate.Aggregate

        Will error if the Aggregation contains duplicate column names
        """
        lookup = {}
        for group, groupby in self.groups.items():
            for agg in self.aggregates:
                for col in agg.get_columns(prefix=self._col_prefix(group)):
                    if col.name in lookup:
                        raise ValueError("Duplicate feature column name found: ", col.name)
                    lookup[col.name] = agg
        return lookup

    def colname_agg_function(self, colname): 
        if colname.endswith('_imp'):
            raise ValueError('Imputation flag columns cannot have their aggregation function inferred')

        aggregate = self.colname_aggregate_lookup[colname] 
        if hasattr(aggregate, 'functions'):
            used_function = next(funcname for funcname in aggregate.functions if colname.endswith(funcname))
            return used_function
        else:
            raise NoAggregateFunctionError()

    def imputation_flag_base(self, colname):
        used_function = self.colname_agg_function(colname)
        if used_function in AGGFUNCS_NEED_MULTIPLE_VALUES:
            return colname
        else:
            return colname.rstrip('_' + used_function)

    def _col_prefix(self, group):
        """
        Helper for creating a column prefix for the group
            group: group clause, for naming columns
        Returns: string for a common column prefix for columns in that group
        """
        return "{prefix}_{group}_".format(prefix=self.prefix, group=group)

    def _get_aggregates_sql(self, group):
        """
        Helper for getting aggregates sql
        Args:
            group: group clause, for naming columns
        Returns: collection of aggregate column SQL strings
        """
        return chain(*[a.get_columns(prefix=self._col_prefix(group)) for a in self.aggregates])

    def get_selects(self):
        """
        Constructs select queries for this aggregation

        Returns: a dictionary of group : queries pairs where
            group are the same keys as groups
            queries is a list of Select queries, one for each date in dates
        """
        queries = {}

        for group, groupby in self.groups.items():
            columns = [make_sql_clause(groupby, ex.text)]
            columns += self._get_aggregates_sql(group)

            gb_clause = make_sql_clause(groupby, ex.literal_column)
            query = ex.select(columns=columns, from_obj=make_sql_clause(self.from_obj, ex.text)).group_by(
                gb_clause
            )

            queries[group] = [query]

        return queries

    def get_imputation_rules(self):
        """
        Constructs a dictionary to lookup an imputation rule from an associated
        column name.

        Returns: a dictionary of column : imputation_rule pairs
        """
        imprules = {}
        for group, groupby in self.groups.items():
            prefix = "{prefix}_{group}_".format(prefix=self.prefix, group=group)
            for a in self.aggregates:
                imprules.update(a.column_imputation_lookup(prefix=prefix))
        return imprules

    def get_table_name(self, group=None, imputed=False):
        """
        Returns name for table for the given group
        """
        if group is None and not imputed:
            name = '"%s_%s"' % (self.prefix, self.suffix)
        elif group is None and imputed:
            name = '"%s_%s_%s"' % (self.prefix, self.suffix, "imputed")
        elif imputed:
            name = '"%s"' % to_sql_name("%s_%s_%s" % (self.prefix, group, "imputed"))
        else:
            name = '"%s"' % to_sql_name("%s_%s" % (self.prefix, group))
        schema = '"%s".' % self.schema if self.schema else ""
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
        return {
            group: CreateTableAs(self.get_table_name(group), next(iter(sels)).limit(0))
            for group, sels in self.get_selects().items()
        }

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
        return {
            group: [InsertFromSelect(self.get_table_name(group), sel) for sel in sels]
            for group, sels in self.get_selects().items()
        }

    def get_drops(self):
        """
        Generate drop queries for this aggregation

        Returns: a dictionary of group : drop pairs where
            group are the same keys as groups
            drop is a raw drop table query for the corresponding table
        """
        return {
            group: "DROP TABLE IF EXISTS %s;" % self.get_table_name(group)
            for group in self.groups
        }

    def get_indexes(self):
        """
        Generate create index queries for this aggregation

        Returns: a dictionary of group : index pairs where
            group are the same keys as groups
            index is a raw create index query for the corresponding table
        """
        return {
            group: "CREATE INDEX ON %s (%s);" % (self.get_table_name(group), groupby)
            for group, groupby in self.groups.items()
        }

    def get_join_table(self):
        """
        Generate a query for a join table
        """
        return ex.Select(
            columns=[make_sql_clause(group, ex.column) for group in self.groups.values()],
            from_obj=self.from_obj
        ).group_by(
            *self.groups.values()
        )

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
            query += "LEFT JOIN %s USING (%s)" % (self.get_table_name(group), groupby)

        return "CREATE TABLE %s AS (%s);" % (self.get_table_name(), query)

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
        if self.schema is not None:
            return "CREATE SCHEMA IF NOT EXISTS %s" % self.schema

    def find_nulls(self, imputed=False):
        """
        Generate query to count number of nulls in each column in the aggregation table

        Returns: a SQL SELECT statement
        """
        query_template = """
            SELECT {cols}
            FROM {state_tbl} t1
            LEFT JOIN {aggs_tbl} t2 USING({group})
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
            state_tbl=self.state_table,
            aggs_tbl=self.get_table_name(imputed=imputed),
            group=self.state_group,
        )

    def _get_impute_select(self, impute_cols, nonimpute_cols, partitionby=None):

        imprules = self.get_imputation_rules()

        # check if we're missing any columns relative to the full set and raise an
        # exception if we are
        missing_cols = set(imprules.keys()) - set(nonimpute_cols + impute_cols)
        if len(missing_cols) > 0:
            raise ValueError("Missing columns in get_impute_create: %s" % missing_cols)

        # key columns and date column
        query = ""

        used_impflags = set()
        # pre-sort and iterate through the combined set to ensure column order
        for col in sorted(nonimpute_cols + impute_cols):
            # just pass through columns that don't require imputation (no nulls found)
            if col in nonimpute_cols:
                query += '\n,"%s"' % col

            # for columns that do require imputation, include SQL to do the imputation work
            # and a flag for whether the value was imputed
            if col in impute_cols:

                # we don't want to add redundant imputation flags. for a given source
                # column and time interval, all of the functions will have identical
                # sets of rows that needed imputation
                # to reliably merge these, we lookup the original aggregate that produced
                # the function, and see its available functions. we expect exactly one of
                # these functions to end the column name and remove it if so
                # this is passed to the imputer
                try:
                    impflag_basecol = self.imputation_flag_base(col)
                except NoAggregateFunctionError:
                    logger.warning("Imputation flag merging is not implemented for "
                                    "AggregateExpression objects that don't define an aggregate "
                                    "function (e.g. composites)")
                    impflag_basecol = col

                impute_rule = imprules[col]

                try:
                    imputer = available_imputations[impute_rule["type"]]
                except KeyError as err:
                    raise ValueError(
                        "Invalid imputation type %s for column %s"
                        % (impute_rule.get("type", ""), col)
                    ) from err

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

    def get_impute_create(self, impute_cols, nonimpute_cols):
        """
        Generates the CREATE TABLE query for the aggregation table with imputation.

        Args:
            impute_cols: a list of column names with null values
            nonimpute_cols: a list of column names without null values

        Returns: a CREATE TABLE AS query
        """

        # key columns and date column
        query = "SELECT %s" % ", ".join(map(str, self.groups.values()))

        # columns with imputation filling as needed
        query += self._get_impute_select(impute_cols, nonimpute_cols)

        # imputation starts from the state table and left joins into the aggregation table
        query += "\nFROM %s t1" % self.state_table
        query += "\nLEFT JOIN %s t2 USING(%s)" % (
            self.get_table_name(),
            self.state_group,
        )

        return "CREATE TABLE %s AS (%s)" % (self.get_table_name(imputed=True), query)

    def execute(self, conn, join_table=None):
        """
        Execute all SQL statements to create final aggregation table.
        Args:
            conn: the SQLAlchemy connection on which to execute
        """
        self.validate(conn)
        create_schema = self.get_create_schema()
        creates = self.get_creates()
        drops = self.get_drops()
        indexes = self.get_indexes()
        inserts = self.get_inserts()
        drop = self.get_drop()
        create = self.get_create(join_table=join_table)

        trans = conn.begin()

        if create_schema is not None:
            conn.execute(create_schema)

        for group in self.groups:
            conn.execute(drops[group])
            conn.execute(creates[group])
            for insert in inserts[group]:
                conn.execute(insert)
            conn.execute(indexes[group])

        # create the aggregation table
        conn.execute(drop)
        conn.execute(create)

        # excute query to find columns with null values and create lists of columns
        # that do and do not need imputation when creating the imputation table
        res = conn.execute(self.find_nulls())
        null_counts = list(zip(res.keys(), res.fetchone()))
        impute_cols = [col for col, val in null_counts if val > 0]
        nonimpute_cols = [col for col, val in null_counts if val == 0]
        res.close()

        # sql to drop and create the imputation table
        drop_imp = self.get_drop(imputed=True)
        create_imp = self.get_impute_create(
            impute_cols=impute_cols, nonimpute_cols=nonimpute_cols
        )

        # create the imputation table
        conn.execute(drop_imp)
        conn.execute(create_imp)

        trans.commit()

    def validate(self, conn):
        """
        Validate the Aggregation to ensure that it will perform as expected.
        This is done against an active SQL connection in order to enable
        validation of the SQL itself.
        """
