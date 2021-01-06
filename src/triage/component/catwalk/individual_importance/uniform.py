from triage.component.catwalk.model_trainers import NO_FEATURE_IMPORTANCE


def _entity_feature_values(matrix, feature_name, as_of_date):
    """Finds the value of the given feature for each entity in a matrix

    Args:
        matrix (pandas.DataFrame), with index either 'entity_id' or 'entity_id'/'as_of_date'
        feature_name (string) The name of a column in the matrix
        as_of_date (datetime.date) This must be one of the valid as_of_dates in the matrix

    Returns: (list) of (entity_id, feature_value) tuples
    """
    results = []
    index_of_entity = matrix.index.names.index("entity_id")
    index_of_date = matrix.index.names.index("as_of_date")
    if feature_name == NO_FEATURE_IMPORTANCE:
        zipped_iter = zip(matrix.index.values, [None] * len(matrix.index.values))
    else:
        zipped_iter = zip(matrix.index.values, matrix[feature_name].tolist())
    for row in zipped_iter:
        index_values, feature_value = row
        entity_id = index_values[index_of_entity]
        if type(index_values[index_of_date].date()) != type(as_of_date):
            raise TypeError("Types of date in matrix and input must match, "
                            f"Matrix was {type(index_values[index_of_date].date())}",
                            f"Input was {type(as_of_date)}")
        if index_values[index_of_date].date() == as_of_date:
            results.append((entity_id, feature_value))
    return results


def uniform_distribution(db_engine, model_id, as_of_date, test_matrix_store, n_ranks):
    """Calculates individual feature importances based on the global feature importances

    Args:
        db_engine (sqlalchemy.engine)
        model_id (int) A model id, expected to be present in triage_metadata.models
        as_of_date (datetime.date) The date to produce individual importances as of
        test_matrix_store (catwalk.storage.MatrixStore) The test matrix
        n_ranks (int) Number of ranks to calculate and save. Defaults to 5

    Returns: (list) dicts with entity_id, feature_value, feature_name, score
    """
    global_feature_importances = [
        row
        for row in db_engine.execute(
            """select feature, feature_importance
        from train_results.feature_importances where model_id = %s
        order by feature_importance desc limit %s""",
            model_id,
            n_ranks,
        )
    ]

    results = []

    for feature_name, feature_importance in global_feature_importances:
        efv = _entity_feature_values(test_matrix_store.design_matrix, feature_name, as_of_date)
        for entity_id, feature_value in efv:
            results.append(
                {
                    "entity_id": entity_id,
                    "feature_value": feature_value,
                    "feature_name": feature_name,
                    "score": feature_importance,
                }
            )
    return results
