class BaseImputation:
    """Base class for various imputation methods
    """

    def __init__(
        self, column, coltype, column_base_for_impflag=None, partitionby=None, null_cat_pattern=None, noflag=False
    ):
        self.column = column
        self.column_base_for_impflag = column_base_for_impflag
        self.coltype = coltype
        self.catcol = coltype in ["categorical", "array_categorical"]
        # categoricals have a null category, so don't need a flag
        self.noflag = True if self.catcol else noflag
        self.partitionby = (
            "" if partitionby is None else "PARTITION BY %s" % partitionby
        )
        # pattern for matching the null category column for a categorical variable
        # (assumes default of __NULL_ from collate.Compare):
        self.null_cat_pattern = (
            "__NULL_" if null_cat_pattern is None else null_cat_pattern
        )

    def _base_sql(self):
        return """COALESCE("{col}", {{imp}}) AS "{col}" """.format(col=self.column)

    def imputed_flag_select_and_alias(self):
        if not self.noflag:
            template = """CASE WHEN "{col}" IS NULL THEN 1::SMALLINT ELSE 0::SMALLINT END""" 
            alias_template = "{base_for_impflag}_imp"
            if self.column_base_for_impflag:
                return (
                    template.format(col=self.column),
                    alias_template.format(base_for_impflag=self.column_base_for_impflag)
                )
            else:
                return (
                    template.format(col=self.column),
                    alias_template.format(base_for_impflag=self.column)
                )

        else:
            # don't need to create a flag for categoricals (since this is handled with the
            # null category) or other imputations that suppress the flag (e.g., zero_noflag)
            return None, None


class ImputeMean(BaseImputation):
    """Class for mean imputation:

    For aggregate features, just take the average of the column (optionally within
    a partition, such as an as_of_date), falling back to 0 if the column is entirely
    null within the partition.

    For categorical features, we flag the NULL category column with a 1 and other
    columns with the mean, again falling back to 0 as necessary
    """

    def __init__(
        self, column, coltype, column_base_for_impflag=None, partitionby=None, null_cat_pattern=None, **kwargs
    ):
        BaseImputation.__init__(
            self,
            column=column,
            coltype=coltype,
            column_base_for_impflag=column_base_for_impflag,
            partitionby=partitionby,
            null_cat_pattern=null_cat_pattern,
        )

    def to_sql(self):
        sql = self._base_sql()

        if not self.catcol:
            # aggregate columm
            return sql.format(
                imp="""AVG("%s") OVER (%s)::REAL, 0::REAL""" % (self.column, self.partitionby)
            )
        elif self.null_cat_pattern in self.column:
            # categorical NULL category
            return sql.format(imp=1)
        else:
            # categorical
            return sql.format(
                imp="""AVG("%s") OVER (%s)::REAL, 0::REAL""" % (self.column, self.partitionby)
            )


class ImputeConstant(BaseImputation):
    """Class for constant value imputation:

    For aggregates, simply fill in the specified value.

    For categoricals, match the value to the column name and fill in the matching
    column with a 1 (as well as the NULL category column)
    """

    def __init__(self, column, coltype, value, column_base_for_impflag=None, null_cat_pattern=None, **kwargs):
        BaseImputation.__init__(
            self,
            column=column,
            coltype=coltype,
            column_base_for_impflag=column_base_for_impflag,
            partitionby=None,
            null_cat_pattern=null_cat_pattern,
        )
        self.value = value

    def to_sql(self):
        sql = self._base_sql()

        if not self.catcol:
            # aggregate column
            return sql.format(imp=self.value)
        else:
            # categorical column
            return sql.format(
                imp="1::SMALLINT"
                if self.value in self.column or self.null_cat_pattern in self.column
                else "0::SMALLINT"
            )


class ImputeZero(BaseImputation):
    """Class for zero filling imputation:

    Fill in the column with a 0, aside from a null category column for categorical
    variables, which is filled with a 1
    """

    def __init__(self, column, coltype, column_base_for_impflag=None, null_cat_pattern=None, **kwargs):
        BaseImputation.__init__(
            self,
            column=column,
            coltype=coltype,
            column_base_for_impflag=column_base_for_impflag,
            partitionby=None,
            null_cat_pattern=null_cat_pattern,
        )

    def to_sql(self):
        sql = self._base_sql()
        return sql.format(
            imp="1::SMALLINT" if self.catcol and self.null_cat_pattern in self.column else ("0::SMALLINT" if self.catcol else "0::REAL")
        )


class ImputeZeroNoFlag(BaseImputation):
    """Class for zero filling with no imputation flag:

    Fill in missing values with 0 without generating an imputation flag. This option
    should be used only for cases where null values are explicitly known to be zero
    such as absence of an entity from an events table indicating that no such event
    has occurred.
    """

    def __init__(self, column, coltype, column_base_for_impflag=None, null_cat_pattern=None, **kwargs):
        BaseImputation.__init__(
            self,
            column=column,
            coltype=coltype,
            column_base_for_impflag=column_base_for_impflag,
            partitionby=None,
            null_cat_pattern=null_cat_pattern,
            noflag=True,
        )

    def to_sql(self):
        sql = self._base_sql()
        return sql.format(imp="0::SMALLINT")


class ImputeNullCategory(BaseImputation):
    """Class for just using the null category for categoricals:

    For a categorical feature, fill the null category with 1, all others with 0
    (essentially the same as ImputeZero for categoricals only)
    """

    def __init__(self, column, coltype, column_base_for_impflag=None, null_cat_pattern=None, **kwargs):
        BaseImputation.__init__(
            self,
            column=column,
            coltype=coltype,
            column_base_for_impflag=column_base_for_impflag,
            partitionby=None,
            null_cat_pattern=null_cat_pattern,
        )
        if not self.catcol:
            raise ValueError(
                "Can only use null category imputation for categorical features"
            )

    def to_sql(self):
        sql = self._base_sql()
        return sql.format(imp="1::SMALLINT" if self.null_cat_pattern in self.column else "0::SMALLINT")


class ImputeBinaryMode(BaseImputation):
    """Class for mode imputation for binaries:

    For a binary variable, fill with a 1 if the mean (potentially in some window)
    is greater than 0.5, 0 otherwise. Note that this is not available for categoricals
    as it does not determine the modal category, just whether a binary is over 50%.
    """

    def __init__(self, column, coltype, column_base_for_impflag=None, partitionby=None, **kwargs):
        BaseImputation.__init__(
            self, column=column, coltype=coltype, partitionby=partitionby,
            column_base_for_impflag=column_base_for_impflag,
        )
        if self.catcol:
            raise ValueError(
                "Can only use binary mode imputation for non-categorical features"
            )

    def to_sql(self):
        sql = self._base_sql()
        return sql.format(
            imp="""CASE WHEN AVG("%s") OVER (%s) > 0.5 THEN 1 ELSE 0 END, 0"""
            % (self.column, self.partitionby)
        )


class ImputeError(BaseImputation):
    """Class for raising an exception on null:

    If you expect your data to be clean, you can use this class to simply generate
    an error if any null values are found in the data rather than continuing with
    an imputation.
    """

    def __init__(self, column, coltype, column_base_for_impflag=None, **kwargs):
        BaseImputation.__init__(self, column=column, coltype=coltype, column_base_for_impflag=column_base_for_impflag)

    def to_sql(self):
        raise ValueError(
            "NULL values found in column with 'error' imputation type: %s" % self.column
        )
