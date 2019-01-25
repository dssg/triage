import logging
from collections import OrderedDict

import sqlalchemy
import sqlparse

from triage.util.conf import convert_str_to_relativedelta
from triage.database_reflection import table_exists

from triage.component.collate import (
    Aggregate,
    Categorical,
    Compare,
    SpacetimeAggregation,
    FromObj
)


class FeatureGenerator(object):
    def __init__(
        self,
        db_engine,
        features_schema_name,
        replace=True,
        feature_start_time=None,
        materialize_subquery_fromobjs=True,
        features_ignore_cohort=False,
    ):
        """Generates aggregate features using collate

        Args:
            db_engine (sqlalchemy.db.engine)
            features_schema_name (string) Name of schema where feature
                tables should be written to
            replace (boolean, optional) Whether or not existing features
                should be replaced
            feature_start_time (string/datetime, optional) point in time before which
                should not be included in features
            features_ignore_cohort (boolean, optional) Whether or not features should be built
                independently of the cohort. Takes longer but means that features can be reused
                for different cohorts.
        """
        self.db_engine = db_engine
        self.features_schema_name = features_schema_name
        self.categorical_cache = {}
        self.replace = replace
        self.feature_start_time = feature_start_time
        self.materialize_subquery_fromobjs = materialize_subquery_fromobjs
        self.features_ignore_cohort = features_ignore_cohort
        self.entity_id_column = "entity_id"
        self.from_objs = {}

    def _validate_keys(self, aggregation_config):
        for key in [
            "from_obj",
            "intervals",
            "groups",
            "knowledge_date_column",
            "prefix",
        ]:
            if key not in aggregation_config:
                raise ValueError(
                    "{} required as key: aggregation config: {}".format(
                        key, aggregation_config
                    )
                )

    def _validate_aggregates(self, aggregation_config):
        if (
            "aggregates" not in aggregation_config
            and "categoricals" not in aggregation_config
            and "array_categoricals" not in aggregation_config
        ):
            raise ValueError(
                "Need either aggregates, categoricals, or array_categoricals"
                + " in {}".format(aggregation_config)
            )

    def _validate_categoricals(self, categoricals):
        for categorical in categoricals:
            if "choice_query" in categorical:
                logging.info("Validating choice query")

                try:
                    with self.db_engine.begin() as conn:
                        conn.execute("explain {}".format(categorical["choice_query"]))
                except Exception as exc:
                    raise ValueError(
                        "choice query does not run. \n"
                        'choice query: "{}"\n'
                        "Full error: {}".format(categorical["choice_query"], exc)
                    )

    def _validate_from_obj(self, from_obj):
        logging.info("Validating from_obj")
        try:
            with self.db_engine.begin() as conn:
                conn.execute("explain select * from {}".format(from_obj))
        except Exception as exc:
            raise ValueError(
                "from_obj query does not run. \n"
                'from_obj: "{}"\n'
                "Full error: {}".format(from_obj, exc)
            )

    def _validate_time_intervals(self, intervals):
        logging.info("Validating time intervals")
        for interval in intervals:
            if interval != "all":
                convert_str_to_relativedelta(interval)

    def _validate_groups(self, groups):
        if "entity_id" not in groups:
            raise ValueError(
                "One of the aggregation groups is required to be entity_id"
            )

    def _validate_imputation_rule(self, aggregate_type, impute_rule):
        """Validate the imputation rule for a given aggregation type."""
        # dictionary of imputation type : required parameters
        valid_imputations = {
            "all": {
                "mean": [],
                "constant": ["value"],
                "zero": [],
                "zero_noflag": [],
                "error": [],
            },
            "aggregates": {"binary_mode": []},
            "categoricals": {"null_category": []},
        }
        valid_imputations["array_categoricals"] = valid_imputations["categoricals"]

        # the valid imputation rules for the specific aggregation type being checked
        valid_types = dict(
            valid_imputations["all"], **valid_imputations[aggregate_type]
        )

        # no imputation rule was specified
        if "type" not in impute_rule.keys():
            raise ValueError("Imputation type must be specified")

        # a rule was specified, but not valid for this type of aggregate
        if impute_rule["type"] not in valid_types.keys():
            raise ValueError(
                "Invalid imputation type %s for %s"
                % (impute_rule["type"], aggregate_type)
            )

        # check that all required parameters exist in the keys of the imputation rule
        required_params = valid_types[impute_rule["type"]]
        for param in required_params:
            if param not in impute_rule.keys():
                raise ValueError(
                    "Missing param %s for %s" % (param, impute_rule["type"])
                )

    def _validate_imputations(self, aggregation_config):
        """Validate the imputation rules in an aggregation config, looping
        through all three types of aggregates. Most of the work here is
        done by _validate_imputation_rule() to check the requirements of
        each imputation rule found
        """
        agg_types = ["aggregates", "categoricals", "array_categoricals"]

        for agg_type in agg_types:
            # base_imp are the top-level rules, `such as aggregates_imputation`
            base_imp = aggregation_config.get(agg_type + "_imputation", {})

            # loop through the individual aggregates
            for agg in aggregation_config.get(agg_type, []):
                # combine any aggregate-level imputation rules with top-level ones
                imp_dict = dict(base_imp, **agg.get("imputation", {}))

                # imputation rules are metric-specific, so check each metric's rule
                for metric in agg["metrics"]:
                    # metric rules may be defined by the metric name (e.g., 'max')
                    # or with the 'all' catch-all, with named metrics taking
                    # precedence. If we fall back to {}, the rule validator will
                    # error out on no metric found.
                    impute_rule = imp_dict.get(metric, imp_dict.get("all", {}))
                    self._validate_imputation_rule(agg_type, impute_rule)

    def _validate_aggregation(self, aggregation_config):
        logging.info("Validating aggregation config %s", aggregation_config)
        self._validate_keys(aggregation_config)
        self._validate_aggregates(aggregation_config)
        self._validate_categoricals(aggregation_config.get("categoricals", []))
        self._validate_from_obj(aggregation_config["from_obj"])
        self._validate_time_intervals(aggregation_config["intervals"])
        self._validate_groups(aggregation_config["groups"])
        self._validate_imputations(aggregation_config)

    def validate(self, feature_aggregation_config):
        """Validate a feature aggregation config applied to this object

        The validations range from basic type checks, key presence checks,
        as well as validating the sql in from objects.

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation

        Raises: ValueError if any part of the config is found to be invalid
        """
        for aggregation in feature_aggregation_config:
            self._validate_aggregation(aggregation)

    def _compute_choices(self, choice_query):
        if choice_query not in self.categorical_cache:
            with self.db_engine.begin() as conn:
                self.categorical_cache[choice_query] = [
                    row[0] for row in conn.execute(choice_query)
                ]

            logging.info(
                "Computed list of categoricals: %s for choice query: %s",
                self.categorical_cache[choice_query],
                choice_query,
            )

        return self.categorical_cache[choice_query]

    def _build_choices(self, categorical):
        logging.info(
            "Building categorical choices for column %s, metrics %s",
            categorical["column"],
            categorical["metrics"],
        )
        if "choices" in categorical:
            logging.info("Found list of configured choices: %s", categorical["choices"])
            return categorical["choices"]
        else:
            return self._compute_choices(categorical["choice_query"])

    def _build_categoricals(self, categorical_config, impute_rules):
        # TODO: only include null flag where necessary
        return [
            Categorical(
                col=categorical["column"],
                choices=self._build_choices(categorical),
                function=categorical["metrics"],
                impute_rules=dict(
                    impute_rules,
                    coltype="categorical",
                    **categorical.get("imputation", {})
                ),
                include_null=True,
                coltype=categorical.get('coltype', None),
            )
            for categorical in categorical_config
        ]

    def _build_array_categoricals(self, categorical_config, impute_rules):
        # TODO: only include null flag where necessary
        return [
            Compare(
                col=categorical["column"],
                op="@>",
                choices={
                    choice: "array['{}'::varchar]".format(choice)
                    for choice in self._build_choices(categorical)
                },
                function=categorical["metrics"],
                impute_rules=dict(
                    impute_rules,
                    coltype="array_categorical",
                    **categorical.get("imputation", {})
                ),
                op_in_name=False,
                quote_choices=False,
                include_null=True,
                coltype=categorical.get('coltype', None)
            )
            for categorical in categorical_config
        ]

    def _aggregation(self, aggregation_config, feature_dates, state_table):
        logging.info(
            "Building collate.SpacetimeAggregation for config %s and %s as_of_dates",
            aggregation_config,
            len(feature_dates),
        )

        # read top-level imputation rules from the aggregation config; we'll allow
        # these to be overridden by imputation rules at the individual feature
        # level as those get parsed as well
        agimp = aggregation_config.get("aggregates_imputation", {})
        catimp = aggregation_config.get("categoricals_imputation", {})
        arrcatimp = aggregation_config.get("array_categoricals_imputation", {})

        aggregates = [
            Aggregate(
                aggregate["quantity"],
                aggregate["metrics"],
                dict(agimp, coltype="aggregate", **aggregate.get("imputation", {})),
                coltype=aggregate.get('coltype', None)
            )
            for aggregate in aggregation_config.get("aggregates", [])
        ]
        logging.info("Found %s quantity aggregates", len(aggregates))
        categoricals = self._build_categoricals(
            aggregation_config.get("categoricals", []), catimp
        )
        logging.info("Found %s categorical aggregates", len(categoricals))
        array_categoricals = self._build_array_categoricals(
            aggregation_config.get("array_categoricals", []), arrcatimp
        )
        logging.info("Found %s array categorical aggregates", len(array_categoricals))
        return SpacetimeAggregation(
            aggregates + categoricals + array_categoricals,
            from_obj=aggregation_config["from_obj"],
            intervals=aggregation_config["intervals"],
            groups=aggregation_config["groups"],
            dates=feature_dates,
            state_table=state_table,
            state_group=self.entity_id_column,
            date_column=aggregation_config["knowledge_date_column"],
            output_date_column="as_of_date",
            input_min_date=self.feature_start_time,
            schema=self.features_schema_name,
            prefix=aggregation_config["prefix"],
            join_with_cohort_table=not self.features_ignore_cohort
        )

    def aggregations(self, feature_aggregation_config, feature_dates, state_table):
        """Creates collate.SpacetimeAggregations from the given arguments

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation
            feature_dates (list) dates to generate features as of
            state_table (string) schema.table_name for state table with all entity/date pairs

        Returns: (list) collate.SpacetimeAggregations
        """
        return [
            self.preprocess_aggregation(
                self._aggregation(aggregation_config, feature_dates, state_table)
            )
            for aggregation_config in feature_aggregation_config
        ]

    def preprocess_aggregation(self, aggregation):
        create_schema = aggregation.get_create_schema()

        if create_schema is not None:
            with self.db_engine.begin() as conn:
                conn.execute(create_schema)

        if self.materialize_subquery_fromobjs:
            # materialize from obj
            from_obj = FromObj(
                from_obj=aggregation.from_obj.text,
                name=f"{aggregation.schema}.{aggregation.prefix}",
                knowledge_date_column=aggregation.date_column
            )
            from_obj.maybe_materialize(self.db_engine)
            aggregation.from_obj = from_obj.table
        return aggregation

    def generate_all_table_tasks(self, aggregations, task_type):
        """Generates SQL commands for creating, populating, and indexing
        feature group tables

        Args:
            aggregations (list) collate.SpacetimeAggregation objects
            type (str) either 'aggregation' or 'imputation'

        Returns: (dict) keys are group table names, values are themselves dicts,
            each with keys for different stages of table creation (prepare, inserts, finalize)
            and with values being lists of SQL commands
        """

        logging.debug("---------------------")

        # pick the method to use for generating tasks depending on whether we're
        # building the aggregations or imputations
        if task_type == "aggregation":
            task_generator = self._generate_agg_table_tasks_for
            logging.debug("---------FEATURE GENERATION------------")
        elif task_type == "imputation":
            task_generator = self._generate_imp_table_tasks_for
            logging.debug("---------FEATURE IMPUTATION------------")
        else:
            raise ValueError("Table task type must be aggregation or imputation")

        logging.debug("---------------------")

        table_tasks = OrderedDict()
        for aggregation in aggregations:
            table_tasks.update(task_generator(aggregation))
        logging.info("Created %s tables", len(table_tasks.keys()))
        return table_tasks

    def create_features_before_imputation(
        self, feature_aggregation_config, feature_dates, state_table=None
    ):
        """Create features before imputation for a set of dates"""
        all_tasks = self.generate_all_table_tasks(
            self.aggregations(
                feature_aggregation_config, feature_dates, state_table=state_table
            ),
            task_type="aggregation",
        )
        logging.info("Generated a total of %s table tasks", len(all_tasks))
        for task_num, task in enumerate(all_tasks.values(), 1):
            prepares = task.get("prepare", [])
            inserts = task.get("inserts", [])
            finalize = task.get("finalize", [])
            logging.info("------------------")
            logging.info("TASK %s ", task_num)
            logging.info(
                "%s prepare queries, %s insert queries, %s finalize queries",
                len(prepares),
                len(inserts),
                len(finalize),
            )
            logging.info("------------------")
            logging.info("")
            logging.info("------------------")
            logging.info("PREPARATION QUERIES")
            logging.info("------------------")
            for query_num, query in enumerate(prepares, 1):
                logging.info("")
                logging.info(
                    "prepare query %s: %s",
                    query_num,
                    sqlparse.format(str(query), reindent=True),
                )
            logging.info("------------------")
            logging.info("INSERT QUERIES")
            logging.info("------------------")
            for query_num, query in enumerate(inserts, 1):
                logging.info("")
                logging.info(
                    "insert query %s: %s",
                    query_num,
                    sqlparse.format(str(query), reindent=True),
                )
            logging.info("------------------")
            logging.info("FINALIZE QUERIES")
            logging.info("------------------")
            for query_num, query in enumerate(finalize, 1):
                logging.info("")
                logging.info(
                    "finalize query %s: %s",
                    query_num,
                    sqlparse.format(str(query), reindent=True),
                )
            self.process_table_task(task)

    def create_all_tables(self, feature_aggregation_config, feature_dates, state_table):
        """Create all feature tables.

        First builds the aggregation tables, and then performs
        imputation on any null values, (requiring a two-step process to
        determine which columns contain nulls after the initial
        aggregation tables are built).

        Args:
            feature_aggregation_config (list) all values, except for
                feature date, necessary to instantiate a
                `collate.SpacetimeAggregation`
            feature_dates (list) dates to generate features as of
            state_table (string) schema.table_name for state table with
                all entity/date pairs

        Returns: (list) table names

        """
        aggs = self.aggregations(feature_aggregation_config, feature_dates, state_table)

        # first, generate and run table tasks for aggregations
        table_tasks_aggregate = self.generate_all_table_tasks(
            aggs, task_type="aggregation"
        )
        self.process_table_tasks(table_tasks_aggregate)

        # second, perform the imputations (this will query the tables
        # constructed above to identify features containing nulls)
        table_tasks_impute = self.generate_all_table_tasks(aggs, task_type="imputation")
        impute_keys = self.process_table_tasks(table_tasks_impute)

        # double-check that the imputation worked and no nulls remain
        # in the data:
        nullcols = []
        with self.db_engine.begin() as conn:
            for agg in aggs:
                results = conn.execute(agg.find_nulls(imputed=True))
                null_counts = results.first().items()
                nullcols += [col for (col, val) in null_counts if val > 0]

        if len(nullcols) > 0:
            raise ValueError(
                "Imputation failed for {} columns. Null values remain in: {}".format(
                    len(nullcols), nullcols
                )
            )

        return impute_keys

    def process_table_task(self, task):
        self.run_commands(task.get("prepare", []))
        self.run_commands(task.get("inserts", []))
        self.run_commands(task.get("finalize", []))

    def process_table_tasks(self, table_tasks):
        for table_name, task in table_tasks.items():
            logging.info("Running feature table queries for %s", table_name)
            self.process_table_task(task)
        return table_tasks.keys()

    def _explain_selects(self, aggregations):
        with self.db_engine.begin() as conn:
            for aggregation in aggregations:
                for selectlist in aggregation.get_selects().values():
                    for select in selectlist:
                        query = "explain " + str(select)
                        results = list(conn.execute(query))
                        logging.debug(str(select))
                        logging.debug(results)

    def _clean_table_name(self, table_name):
        # remove the schema and quotes from the name
        return table_name.split(".")[1].replace('"', "")

    def _table_exists(self, table_name):
        try:
            with self.db_engine.begin() as conn:
                conn.execute(
                    "select 1 from {}.{} limit 1".format(
                        self.features_schema_name, table_name
                    )
                ).first()
        except sqlalchemy.exc.ProgrammingError:
            return False
        else:
            return True

    def run_commands(self, command_list):
        with self.db_engine.begin() as conn:
            for command in command_list:
                logging.debug("Executing feature generation query: %s", command)
                conn.execute(command)

    def _aggregation_index_query(self, aggregation, imputed=False):
        return "CREATE INDEX ON {} ({}, {})".format(
            aggregation.get_table_name(imputed=imputed),
            self.entity_id_column,
            aggregation.output_date_column,
        )

    def _aggregation_index_columns(self, aggregation):
        return sorted(
            [group for group in aggregation.groups.keys()]
            + [aggregation.output_date_column]
        )

    def index_column_lookup(self, aggregations, imputed=True):
        return dict(
            (
                self._clean_table_name(aggregation.get_table_name(imputed=imputed)),
                self._aggregation_index_columns(aggregation),
            )
            for aggregation in aggregations
        )

    def _needs_features(self, aggregation):
        imputed_table = self._clean_table_name(
            aggregation.get_table_name(imputed=True)
        )

        if self._table_exists(imputed_table):
            check_query = (
                f"select 1 from {aggregation.state_table} "
                f"left join {self.features_schema_name}.{imputed_table} "
                "using (entity_id, as_of_date) "
                f"where {self.features_schema_name}.{imputed_table}.entity_id is null limit 1"
            )
            if self.db_engine.execute(check_query).scalar():
                logging.warning(
                    "Imputed feature table %s did not contain rows from the "
                    "entire cohort, need to rebuild features", imputed_table)
                return True
        else:
            logging.warning("Imputed feature table %s did not exist, "
                            "need to build features", imputed_table)
            return True
        logging.warning("Imputed feature table %s looks good, "
                        "skipping feature building!", imputed_table)
        return False

    def _generate_agg_table_tasks_for(self, aggregation):
        """Generates SQL commands for preparing, populating, and finalizing
        each feature group table in the given aggregation

        Args:
            aggregation (collate.SpacetimeAggregation)

        Returns: (dict) of structure {
            'prepare': list of commands to prepare table for population
            'inserts': list of commands to populate table
            'finalize': list of commands to finalize table after population
        }
        """
        creates = aggregation.get_creates()
        drops = aggregation.get_drops()
        indexes = aggregation.get_indexes()
        inserts = aggregation.get_inserts()
        table_tasks = OrderedDict()
        for group in aggregation.groups:
            group_table = self._clean_table_name(
                aggregation.get_table_name(group=group)
            )
            if self.replace or self._needs_features(aggregation):
                table_tasks[group_table] = {
                    "prepare": [drops[group], creates[group]],
                    "inserts": inserts[group],
                    "finalize": [indexes[group]],
                }
                logging.info("Created table tasks for %s", group_table)
            else:
                logging.info("Skipping feature table creation for %s", group_table)
                table_tasks[group_table] = {}
        logging.info("Created table tasks for aggregation")
        if self.replace or self._needs_features(aggregation):
            table_tasks[self._clean_table_name(aggregation.get_table_name())] = {
                "prepare": [aggregation.get_drop(), aggregation.get_create()],
                "inserts": [],
                "finalize": [self._aggregation_index_query(aggregation)],
            }
        else:
            table_tasks[self._clean_table_name(aggregation.get_table_name())] = {}

        return table_tasks

    def _generate_imp_table_tasks_for(self, aggregation, drop_preagg=True):
        """Generate SQL statements for preparing, populating, and
        finalizing imputations, for each feature group table in the
        given aggregation.

        Requires the existance of the underlying feature and aggregation
        tables defined in `_generate_agg_table_tasks_for()`.

        Args:
            aggregation (collate.SpacetimeAggregation)
            drop_preagg: boolean to specify dropping pre-imputation
                tables

        Returns: (dict) of structure {
                'prepare': list of commands to prepare table for population
                'inserts': list of commands to populate table
                'finalize': list of commands to finalize table after population
            }

        """
        table_tasks = OrderedDict()
        imp_tbl_name = self._clean_table_name(aggregation.get_table_name(imputed=True))

        if not self.replace and not self._needs_features(aggregation):
            logging.warning("Skipping imputation table creation for %s", imp_tbl_name)
            table_tasks[imp_tbl_name] = {}
            return table_tasks

        if not aggregation.state_table:
            logging.warning(
                "No state table defined in aggregation, cannot create imputation table for %s",
                imp_tbl_name,
            )
            table_tasks[imp_tbl_name] = {}
            return table_tasks

        if not table_exists(aggregation.state_table, self.db_engine):
            logging.warning(
                "State table %s does not exist, cannot create imputation table for %s",
                aggregation.state_table,
                imp_tbl_name,
            )
            table_tasks[imp_tbl_name] = {}
            return table_tasks

        # excute query to find columns with null values and create lists of columns
        # that do and do not need imputation when creating the imputation table
        with self.db_engine.begin() as conn:
            results = conn.execute(aggregation.find_nulls())
            null_counts = results.first().items()
        impute_cols = [col for (col, val) in null_counts if val > 0]
        nonimpute_cols = [col for (col, val) in null_counts if val == 0]

        # table tasks for imputed aggregation table, most of the work is done here
        # by collate's get_impute_create()
        table_tasks[imp_tbl_name] = {
            "prepare": [
                aggregation.get_drop(imputed=True),
                aggregation.get_impute_create(
                    impute_cols=impute_cols, nonimpute_cols=nonimpute_cols
                ),
            ],
            "inserts": [],
            "finalize": [self._aggregation_index_query(aggregation, imputed=True)],
        }
        logging.info("Created table tasks for imputation: %s", imp_tbl_name)

        # do some cleanup:
        # drop the group-level and aggregation tables, just leaving the
        # imputation table if drop_preagg=True
        if drop_preagg:
            drops = aggregation.get_drops()
            table_tasks[imp_tbl_name]["finalize"] += list(drops.values()) + [
                aggregation.get_drop()
            ]
            logging.info("Added drop table cleanup tasks: %s", imp_tbl_name)

        return table_tasks
