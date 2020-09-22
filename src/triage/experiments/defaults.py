import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import os
import yaml

def fill_timechop_config_missing(config, db_engine):
    """
    Fill with default values the temporal_config params if they are missing

    Args:
        config (dict) a triage experiment configuration
        db_engine (psycopg2 connection) a db connection

    Returns: (dict) a triage temporal_config
    """
    timechop_config = config['temporal_config']

    default_config = {'model_update_frequency': '100y',
                      'training_as_of_date_frequencies': '100y',
                      'test_as_of_date_frequencies': '100y',
                      'max_training_histories': '0d',
                      'test_durations': '0d',
                      }

    # Checks if label_timespan is present
    if 'label_timespans' in timechop_config.keys():
        if any([k in timechop_config.keys() for k in ['training_label_timespans', 'test_labels_timespans']]):
            raise KeyError("You can't always get what you want, but just sometimes, you get what you need: The config file has conflicting keys: the 'label_timespan' and 'training_label_timespans' and/or 'test_label_timespans'")
        default_config['training_label_timespans'] = default_config['test_label_timespans'] = timechop_config['label_timespans']
        timechop_config.pop('label_timespans') ## We don't need this value anymore

    # Checks if some of the date range  limits  is missing, if so replaces with
    # min, max accordingy from de from_objs
    if any([k not in timechop_config.keys() for k in ['feature_start_time', 'feature_end_time', 'label_start_time', 'label_end_time']]):
        from_query = "(select min({knowledge_date}) as min_date, max({knowledge_date}) as max_date from (select * from {from_obj}) as t)"

        feature_aggregations = config['feature_aggregations']

        from_queries = [from_query.format(knowledge_date = agg['knowledge_date_column'], from_obj=agg['from_obj']) for agg in feature_aggregations]

        unions = "\n union \n".join(from_queries)

        query = "select to_char(min(min_date), 'YYYY-MM-DD'), to_char(max(max_date), 'YYYY-MM-DD') from ({unions}) as u".format(unions=unions)

        with db_engine.connect() as conn:
            rs = conn.execute(query)
            min_date, max_date = rs.fetchall()[0]

        default_config['feature_start_time'] = default_config['label_start_time'] = min_date
        default_config['feature_end_time'] = default_config['label_end_time'] = max_date

    # Replaces missing values
    default_config.update(timechop_config)

    return default_config


def fill_cohort_config_missing(config):
    """
    If none cohort_config section is provided, include all the entities by default

    Args:
        config (dict) a triage experiment configuration

    Returns: (dict) a triage cohort config
    """
    from_query = "(select entity_id, {knowledge_date} as knowledge_date from (select * from {from_obj}) as t)"

    feature_aggregations = config['feature_aggregations']

    from_queries = [from_query.format(knowledge_date = agg['knowledge_date_column'], from_obj=agg['from_obj']) for agg in feature_aggregations]

    unions = "\n union \n".join(from_queries)

    query = f"select distinct entity_id from ({unions}) as e" +" where knowledge_date < '{as_of_date}'"

    cohort_config = config.get('cohort_config', {})
    default_config = {'query': query, 'name': 'all_entities'}

    default_config.update(cohort_config)

    return default_config


def fill_feature_group_definition(config):
    """
    If feature_group_definition is not presents, this function sets it to all
    the distinct feature_aggregations' prefixes

    Args:
        config (dict) a triage experiment configuration

    Returns: (dict) a triage feature_group config
    """
    feature_group_definition = config.get('feature_group_definition', {})
    if not feature_group_definition:
        feature_aggregations = config['feature_aggregations']

        feature_group_definition['prefix'] = list({agg['prefix'] for agg in feature_aggregations})

    return feature_group_definition


def fill_model_grid_presets(config):
    """Determine if model grid preset is being used and return the appropriate grid if so

       Args:
            config (dict) a triage experiment configuration

        Returns: (dict) a triage model grid config
    """

    grid_config = config.get('grid_config')
    preset_type = config.get('model_grid_preset')

    if preset_type is not None:
        if grid_config is not None:
            # if you specified both grid_config and model_grid_preset, you're doing something wrong
            raise KeyError("There can only be one (cannot specify both model_grid_preset and grid_config)")
        grid_config = model_grid_preset(preset_type)

    return grid_config


def model_grid_preset(grid_type):
    """Load a preset model grid.

       Args:
            grid_type (string) The type of preset grid to load. May
                by `quickstart`, `small`, `medium`, `large`, or `texas`

        Returns: (dict) a triage model grid config
    """

    presets_file = os.path.join(os.path.dirname(__file__), 'model_grid_presets.yaml')
    with open(presets_file, 'r') as f:
        model_grid_presets = yaml.full_load(f)

    # output is a collector for the resulting grid, so initialize it with the parameters
    # at the level of preset grid we want and find the next-lowest level to incorporate
    output = model_grid_presets[grid_type]['grid'].copy()
    prev_type = model_grid_presets[grid_type]['prev']

    # collapse the grid parameters down the levels until we reach one with no lower level
    while prev_type is not None:
        prev = model_grid_presets[prev_type]['grid'].copy()

        # look for new model types and hyperparameters to incorporate into the output
        for model_type in set(output.keys()).union(set(prev.keys())):
            curr_model = output.get(model_type, {}).copy()
            # if the model type exists in the lower-level preset, update any associated hyperparameter
            # values in the output (those only in the higher level grid will pass through unchanged)
            for hyperparam in prev.get(model_type, {}).keys():
                curr_model[hyperparam] = sorted(list(set(curr_model.get(hyperparam, []) + prev[model_type][hyperparam])), key=lambda x: x if x is not None else 0)
            output[model_type] = curr_model

        # traverse the linked list to one level deeper and repeat
        prev_type = model_grid_presets[prev_type]['prev']

    return output
