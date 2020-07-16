import json
import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

DEFAULT_KEYS = [
    "label_timespan",
    "label_name",
    "as_of_date_frequency",
    "max_training_history",
    "state",
    "cohort_name",
    "feature_groups",
]


class ModelGrouper:
    """Assign a model group id to given model input based on default or custom configuration

    This class is a wrapper around the `get_model_group_id` stored procedure,
    which interfaces with the model_groups table provision a stable model group id .
    The role of this class is mainly to provide data conversion, sensible defaults, and
    an abstraction layer over the database.

    Args:
        model_group_keys (list) A list of matrix metadata keys to uniquely define a model group.'
            In addition, the non-matrix attributes 'class_path' and 'parameters', referring to
            classifier training arguments, can be sent.
    """

    def __init__(self, model_group_keys=()):
        self.model_group_keys = frozenset(model_group_keys)

    def _final_model_group_args(self, class_path, parameters, matrix_metadata):
        """Generates model grouping arguments based on input.

        Applies a set of default or custom grouping keys depending on the object's
        configuration.

        Formats output in the structure recognized by the get_model_group_id stored procedure:
        {
            'class_path: (string)
            'parameters': (dict)
            'feature_names': (list)
            'model_config': (dict)
        }

        Args:
        class_path (string): a full class path for the classifier
        parameters (dict): all hyperparameters to be passed to the classifier
        matrix_metadata(dict): key-value pairs describing the configuration that produced
            a matrix used for training

        Returns: (dict) a dictionary of arguments suitable for the 'get_model_group_id'
            stored procedure
        """
        # step 1: is there an override?
        if len(self.model_group_keys) > 0:
            final = {}
            keys = set(self.model_group_keys)
            if "class_path" in keys:
                final["class_path"] = class_path
                keys.remove("class_path")
            else:
                final["class_path"] = ""

            if "parameters" in keys:
                final["parameters"] = parameters
                keys.remove("parameters")
            else:
                final["parameters"] = {}

            if "feature_names" in keys:
                final["feature_names"] = matrix_metadata["feature_names"]
                keys.remove("feature_names")
            else:
                final["feature_names"] = []

            final["model_config"] = {}

            for model_group_key in keys:
                final["model_config"][model_group_key] = matrix_metadata[
                    model_group_key
                ]

            return final

        # step 2: if no override, apply defaults
        else:
            model_config = {}
            for model_group_key in DEFAULT_KEYS:
                model_config[model_group_key] = matrix_metadata[model_group_key]

            return dict(
                class_path=class_path,
                parameters=parameters,
                feature_names=matrix_metadata["feature_names"],
                model_config=model_config,
            )

    def get_model_group_id(self, class_path, parameters, matrix_metadata, db_engine):
        """
        Returns model group id using store procedure 'get_model_group_id' which will
        return the same value for models with the same class_path, parameters,
        features, and model_config

        Args:
            class_path (string) A full classpath to the model class
            parameters (dict) hyperparameters to give to the model constructor
            matrix_metadata (dict) stored metadata about the train matrix
            db_engine (sqlalchemy.engine) A database engine pointing to a database with
             a results.model_groups table and get_model_group_id stored procedure

        Returns: (int) a database id for the model group id
        """
        model_group_args = self._final_model_group_args(
            class_path, parameters, matrix_metadata
        )
        db_conn = db_engine.raw_connection()
        cur = db_conn.cursor()
        cur.execute(
            "SELECT EXISTS ( "
            "       SELECT * "
            "       FROM pg_catalog.pg_proc "
            "       WHERE proname = 'get_model_group_id' ) "
        )
        condition = cur.fetchone()

        if condition:
            query = (
                "SELECT get_model_group_id( "
                "            '{class_path}'::TEXT, "
                "            '{parameters}'::JSONB, "
                "             ARRAY{feature_names}::TEXT [] , "
                "            '{model_config}'::JSONB )".format(
                    class_path=model_group_args["class_path"],
                    parameters=json.dumps(model_group_args["parameters"]),
                    feature_names=list(model_group_args["feature_names"]),
                    model_config=json.dumps(
                        model_group_args["model_config"], sort_keys=True
                    ),
                )
            )
            logger.spam(f"Getting model group from query {query}")
            cur.execute(query)
            db_conn.commit()
            model_group_id = cur.fetchone()
            model_group_id = model_group_id[0]

        else:
            logger.warning("Could not found stored procedure public.model_group_id")
            model_group_id = None
        db_conn.close()

        return model_group_id
