from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import pandas as pd


def get_engine(dbname, user, host, port, password):
    """
    Get SQLalchemy engine using credentials.

    Input:
    db: database name
    user: Username
    host: Hostname of the database server
    port: Port number
    passwd: Password for the database
    """

    url = 'postgresql://{user}:{passwd}@{host}:{port}/{db}'.format(
        user=user, passwd=password, host=host, port=port, db=dbname)
    engine = create_engine(url, poolclass=NullPool)
    return engine


def get_prediction_score(conn, model_id, as_of_date):
    """
    This methods retrieves the score and label value for a given model_id and as_of_date
    """
    if type(model_id) != tuple:
        model_id = tuple([model_id])

    query_score_label = """
    SELECT
      model_id,
      entity_id,
      score,
      coalesce(rank_abs, rank()  OVER (ORDER BY score DESC )) AS rank_abs,
      label_value
    FROM results.predictions
    WHERE model_id IN %(model_id)s AND as_of_date = %(as_of_date)s
    """

    df_score_label = pd.read_sql(
        query_score_label,
        con=conn,
        params={'model_id': model_id,
                'as_of_date': as_of_date}
    )

    return df_score_label


def get_train_test_matrix_model_uuid(conn, model_id, as_of_date):
    """
    This methods retrieves train and test matrix uuid as well as the model hash for a given model_id and as_of_date
    """

    query_topn_features = """
    SELECT
      train_matrix_uuid::TEXT,
      matrix_uuid::TEXT AS test_matrix_uuid,
      model_hash::TEXT
    FROM results.models
      JOIN results.predictions USING (model_id)
    WHERE 
        model_id = %(model_id)s AND 
        as_of_date = %(as_of_date)s
    GROUP BY train_matrix_uuid, test_matrix_uuid, model_hash
    """

    df_uuid = pd.read_sql(
        query_topn_features,
        con=conn,
        params={'model_id': model_id,
                'as_of_date': as_of_date}
    )

    return [df_uuid.train_matrix_uuid.tolist()[0], df_uuid.test_matrix_uuid.tolist()[0], df_uuid.model_hash.tolist()[0]]


def get_features_importances(conn, model_ids):
    """
    This methods retrieves the topn features according to the stored feature importance ordered by rank absolute for a given model_id and as_of_date
    """

    if type(model_ids) != tuple:
        model_ids = tuple([model_ids])

    query_topn_features_imp = """
    SELECT
        model_id,
        feature,
        feature_importance,
        rank_abs
    FROM results.feature_importances
    WHERE model_id IN %(model_id)s
    """

    df_features = pd.read_sql(
        query_topn_features_imp,
        con=conn,
        params={'model_id': model_ids}
    )

    return df_features


def get_topn_features(conn, model_id, topn):
    """
    This methods retrieves the topn features according to the stored feature importance ordered by rank absolute for a given model_id and as_of_date
    """

    query_topn_features = """
    SELECT feature
    FROM results.feature_importances
    WHERE model_id = %(model_id)s AND
          rank_abs <= %(topn)s
    ORDER BY rank_abs
    """

    df_topk_features = pd.read_sql(
        query_topn_features,
        con=conn,
        params={'model_id': model_id,
                'topn': topn}
    )

    return df_topk_features.feature.tolist()


# get top_k officers
def get_topk_entities(conn, model_id, as_of_date, topk):
    """
    This methods retrieves the topk  entities according to the stored score ordered by rank  for a given model_id and as_of_date
    """

    query_topk_entities = """
    SELECT 
          entity_id
    FROM results.predictions
    WHERE predictions.model_id = %(model_id)s AND 
          as_of_date = %(as_of_date)s 
    ORDER BY score DESC
    LIMIT %(topk)s
    """

    df_topk_entities = pd.read_sql(
        query_topk_entities,
        con=conn,
        params={'model_id': model_id,
                'as_of_date': as_of_date,
                'topk': topk}
    )
    df_topk_entities = df_topk_entities.set_index('entity_id')

    return df_topk_entities
