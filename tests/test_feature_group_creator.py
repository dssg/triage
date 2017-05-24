from architect.features import FeatureGroupCreator


def test_table_group():
    group = FeatureGroupCreator(
        definition={'tables': ['one', 'three']}
    )

    assert group.subsets({
        'one': ['col_a', 'col_b', 'col_c'],
        'two': ['col_a', 'col_b', 'col_c'],
        'three': ['col_a', 'col_b', 'col_c'],
    }) == [
        {'one': ['col_a', 'col_b', 'col_c']},
        {'three': ['col_a', 'col_b', 'col_c']},
    ]


def test_prefix_group():
    group = FeatureGroupCreator(
        definition={'prefix': ['major', 'severe']}
    )

    assert group.subsets({
        'one': ['minor_a', 'minor_b', 'minor_c'],
        'two': ['severe_a', 'severe_b', 'severe_c'],
        'three': ['major_a', 'major_b', 'major_c'],
        'four': ['minor_a', 'minor_b', 'minor_c'],
    }) == [
        {'three': ['major_a', 'major_b', 'major_c']},
        {'two': ['severe_a', 'severe_b', 'severe_c']},
    ]


def test_multiple_criteria():
    group = FeatureGroupCreator(
        definition={
            'prefix': ['major', 'severe'],
            'tables': ['one', 'two'],
        }
    )

    assert group.subsets({
        'one': ['minor_a', 'minor_b', 'minor_c'],
        'two': ['severe_a', 'severe_b', 'severe_c'],
        'three': ['major_a', 'major_b', 'major_c'],
        'four': ['minor_a', 'minor_b', 'minor_c'],
    }) == [
        {'three': ['major_a', 'major_b', 'major_c']},
        {'two': ['severe_a', 'severe_b', 'severe_c']},
        {'one': ['minor_a', 'minor_b', 'minor_c']},
        {'two': ['severe_a', 'severe_b', 'severe_c']},
    ]
