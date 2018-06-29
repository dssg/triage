#!/usr/bin/env python3
"""
CLI Interface Post Model
"""
from utils import get_conn_engine
import sys
from plot_functions import (plot_score_dist_classes,
                            plot_precision_recall_n)


if __name__ == "__main__":

    db_file = sys.argv[1]
    model_group_id = sys.argv[2]

    engine = get_conn_engine(db_file)

    query = """SELECT
                   distinct model_id
               FROM
                   results.models
               WHERE
                   model_group_id={}
            """.format(model_group_id)

    model_ids = pd.read_sql(query, engine)['model_id'].values

    for model_id in model_ids:
        plot_score_dist_classes(model_id, engine)
        plot_precision_recall_n(model_id, engine, output_type='save')
