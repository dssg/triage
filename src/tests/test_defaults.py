from sqlalchemy import create_engine
import testing.postgresql
import pytest

from triage.component.catwalk.db import ensure_db

from tests.utils import sample_config, populate_source_data
from triage.experiments.defaults import (
    fill_timechop_config_missing,
    fill_feature_group_definition,
    fill_model_grid_presets,
    model_grid_preset,
)


def test_fill_feature_group_definition():
    config = sample_config()
    fg_definition = fill_feature_group_definition(config)
    assert sorted(fg_definition['all']) == [True]


def test_fill_timechop_config_missing():
    remove_keys = [
        'model_update_frequency',
        'training_as_of_date_frequencies',
        'test_as_of_date_frequencies',
        'max_training_histories',
        'test_durations',
        'feature_start_time',
        'feature_end_time',
        'label_start_time',
        'label_end_time',
        'training_label_timespans',
        'test_label_timespans'
        ]

    # ensure redundant keys properly raise errors
    config = sample_config()
    config['temporal_config']['label_timespans'] = '1y'
    with pytest.raises(KeyError):
        timechop_config = fill_timechop_config_missing(config, None)

    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        populate_source_data(db_engine)
        config = sample_config()

        for key in remove_keys:
            config['temporal_config'].pop(key)
        config['temporal_config']['label_timespans'] = '1y'

        timechop_config = fill_timechop_config_missing(config, db_engine)

        assert timechop_config['model_update_frequency'] == '100y'
        assert timechop_config['training_as_of_date_frequencies'] == '100y'
        assert timechop_config['test_as_of_date_frequencies'] == '100y'
        assert timechop_config['max_training_histories'] == '0d'
        assert timechop_config['test_durations'] == '0d'
        assert timechop_config['training_label_timespans'] == '1y'
        assert timechop_config['test_label_timespans'] == '1y'
        assert 'label_timespans' not in timechop_config.keys()
        assert timechop_config['feature_start_time'] == '2010-10-01'
        assert timechop_config['feature_end_time'] == '2013-10-01'
        assert timechop_config['label_start_time'] == '2010-10-01'
        assert timechop_config['label_end_time'] == '2013-10-01'


def test_model_grid_preset():
    test_cases = {
        'quickstart': {'total_items': 8, 'model_types': 3},
        'small': {'total_items': 35, 'model_types': 4},
        'medium': {'total_items': 64, 'model_types': 6},
        'large': {'total_items': 83, 'model_types': 6},
        'texas': {'total_items': 115, 'model_types': 9},
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
    fill_grid = fill_model_grid_presets(config)
    assert len(fill_grid) == 3
    assert len(fill_grid.get('sklearn.tree.DecisionTreeClassifier', {}).get('max_depth', [])) == 3
    assert len(fill_grid.get('sklearn.tree.DecisionTreeClassifier', {}).get('criterion', [])) == 1
