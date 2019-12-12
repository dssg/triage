from sqlalchemy import create_engine
import testing.postgresql
import pytest

from triage.component.catwalk.db import ensure_db

from tests.utils import sample_config, populate_source_data
from triage.experiments.defaults import (
    fill_timechop_config_missing,
    fill_cohort_config_missing,
    fill_feature_group_definition,
    fill_model_grid_presets,
    model_grid_preset,
)


def test_fill_cohort_config_missing():
    config = sample_config()
    config.pop('cohort_config')
    cohort_config = fill_cohort_config_missing(config)
    assert cohort_config == {
        'query': "select distinct entity_id from "
                "((select entity_id, as_of_date as knowledge_date from "
                "(select * from cat_complaints) as t)\n union \n(select entity_id, "
                "as_of_date as knowledge_date from (select * from entity_zip_codes "
                "join zip_code_events using (zip_code)) as t)) as e "
                "where knowledge_date < '{as_of_date}'",
        'name': 'all_entities'
        }


def test_fill_feature_group_definition():
    config = sample_config()
    fg_definition = fill_feature_group_definition(config)
    assert sorted(fg_definition['prefix']) == ['entity_features', 'zip_code_features']


def test_fill_timechop_config_missing():
    pass


def test_model_grid_preset():
    test_cases = {
        'quickstart': {'total_items': 7, 'model_types': 3},
        'small': {'total_items': 34, 'model_types': 4},
        'medium': {'total_items': 63, 'model_types': 6},
        'large': {'total_items': 82, 'model_types': 6},
        'texas': {'total_items': 114, 'model_types': 9},
    }

    for grid_type, exp_results in test_cases.items():
        preset_grid = model_grid_preset(grid_type)
        assert len(preset_grid) == exp_results['model_types']
        total_items = sum([len(y) for x in preset_grid.values() for y in x.values()])
        assert total_items == exp_results['total_items']


def test_fill_model_grid_presets():

    # case 1: has grid, no preset
    config = sample_config()
    fill_grid = fill_model_grid_presets(config)
    assert fill_grid == config['grid_config']

    # case 2: has preset, no grid
    config = sample_config()
    config.pop('grid_config')
    config['model_grid_preset'] = 'quickstart'
    fill_grid = fill_model_grid_presets(config)
    assert len(fill_grid) == 3

    # case 3: neither
    config = sample_config()
    config.pop('grid_config')
    fill_grid = fill_model_grid_presets(config)
    assert fill_grid is None

    # case 4: both
    config = sample_config()
    config['model_grid_preset'] = 'quickstart'
    with pytest.raises(KeyError):
        fill_grid = fill_model_grid_presets(config)
