import itertools

from architect.feature_group_mixer import FeatureGroupMixer


def test_feature_group_mixer_leave_one_out():
    english_numbers = {'one': ['two', 'three'], 'four': ['five', 'six']}
    letters = {'a': ['b', 'c'], 'd': ['e', 'f']}
    german_numbers = {'eins': ['zwei', 'drei'], 'vier': ['funf', 'sechs']}
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(['leave-one-out']).generate(feature_groups)
    expected = [
        dict(itertools.chain(letters.items(), german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), letters.items())),
    ]
    assert result == expected


def test_feature_group_mixer_leave_one_in():
    english_numbers = {'one': ['two', 'three'], 'four': ['five', 'six']}
    letters = {'a': ['b', 'c'], 'd': ['e', 'f']}
    german_numbers = {'eins': ['zwei', 'drei'], 'vier': ['funf', 'sechs']}
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(['leave-one-in']).generate(feature_groups)
    expected = [
        english_numbers,
        letters,
        german_numbers
    ]
    assert result == expected


def test_feature_group_mixer_all():
    english_numbers = {'one': ['two', 'three'], 'four': ['five', 'six']}
    letters = {'a': ['b', 'c'], 'd': ['e', 'f']}
    german_numbers = {'eins': ['zwei', 'drei'], 'vier': ['funf', 'sechs']}
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(['all']).generate(feature_groups)
    expected = [
        dict(itertools.chain(english_numbers.items(), letters.items(), german_numbers.items())),
    ]
    assert result == expected
