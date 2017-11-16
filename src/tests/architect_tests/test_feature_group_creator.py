from architect.features import FeatureGroupCreator
from unittest import TestCase


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
    # ensure we test prefixes with underscores
    group = FeatureGroupCreator(
        definition={'prefix': ['major_viol', 'severe_viol']}
    )

    assert group.subsets({
        'one': ['minor_viol_a', 'minor_viol_b', 'minor_viol_c'],
        'two': ['severe_viol_a', 'severe_viol_b', 'severe_viol_c'],
        'three': ['major_viol_a', 'major_viol_b', 'major_viol_c'],
        'four': ['minor_viol_a', 'minor_viol_b', 'minor_viol_c'],
    }) == [
        {'three': ['major_viol_a', 'major_viol_b', 'major_viol_c']},
        {'two': ['severe_viol_a', 'severe_viol_b', 'severe_viol_c']},
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


class TestValidations(TestCase):
    def test_not_a_dict(self):
        creator = FeatureGroupCreator(definition='prefix')
        with self.assertRaises(ValueError):
            creator.validate()

    def test_badsubsetter(self):
        creator = FeatureGroupCreator(definition={
            'prefix': ['major', 'severe'],
            'other': ['other'],
        })
        with self.assertRaises(ValueError):
            creator.validate()

    def test_stringvalue(self):
        creator = FeatureGroupCreator(definition={
            'prefix': 'major',
        })
        with self.assertRaises(ValueError):
            creator.validate()

    def test_goodconfig(self):
        creator = FeatureGroupCreator(definition={
            'prefix': ['major', 'severe'],
            'tables': ['one', 'two'],
        })
        creator.validate()
