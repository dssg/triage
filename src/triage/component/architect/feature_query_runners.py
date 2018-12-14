import logging

import sqlparse

from triage.database_reflection import table_exists


def run_statements(statement_list, db_engine):
    with db_engine.begin() as conn:
        for statement in statement_list:
            logging.debug("Executing feature generation query: %s", statement)
            conn.execute(statement)


def process_table_task(task, db_engine):
    run_statements(task.get("prepare", []), db_engine)
    run_statements(task.get("inserts", []), db_engine)
    run_statements(task.get("finalize", []), db_engine)


def process_table_tasks(table_tasks, db_engine, verbose=False):
    for task, task_num in enumerate(table_tasks, 1):
        if verbose:
            log_verbose_task_info(task, task_num)
        process_table_task(task, db_engine)


def needs_features(feature_block, db_engine):
    imputed_table = feature_block.get_final_feature_table_name()

    if table_exists(imputed_table, db_engine):
        check_query = (
            f"select 1 from {feature_block.cohort_table} "
            f"left join {imputed_table} "
            "using (entity_id, as_of_date) "
            f"where {imputed_table}.entity_id is null limit 1"
        )
        if db_engine.execute(check_query).scalar():
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


def generate_preimpute_tasks(feature_blocks, replace, db_engine):
    table_tasks = []
    for block in feature_blocks:
        if replace or needs_features(block, db_engine):
            table_tasks.append({
                "prepare": block.get_preinsert_queries(),
                "inserts": block.get_inserts(),
                "finalize": block.get_postinsert_queries()
            })
            logging.info("Generated tasks to create %s feature block tables", len(table_tasks))
        else:
            logging.info("Skipping feature table creation for %s", block)
    return table_tasks


def generate_impute_tasks(feature_blocks, replace, db_engine):
    table_tasks = []
    for block in feature_blocks:
        if replace or needs_features(block, db_engine):
            table_tasks.append({
                "prepare": block.get_impute_queries(),
                "inserts": [],
                "finalize": []
            })
            logging.info("Generated tasks to create %s feature block tables", len(table_tasks))
        else:
            logging.info("Skipping feature table creation for %s", block)
    return table_tasks


def create_all_tables(feature_blocks, replace, db_engine):
    """Create all feature tables.

    First builds the aggregation tables, and then performs
    imputation on any null values, (requiring a two-step process to
    determine which columns contain nulls after the initial
    aggregation tables are built).
    """
    process_table_tasks(generate_preimpute_tasks(feature_blocks, replace, db_engine))
    process_table_tasks(generate_impute_tasks(feature_blocks, replace, db_engine))

    # perform a sanity check that no nulls were left after imputation
    for feature_block in feature_blocks:
        feature_block.verify_no_nulls()


def log_verbose_task_info(task, task_num):
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
