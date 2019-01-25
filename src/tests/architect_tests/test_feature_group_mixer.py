import itertools

from triage.component.architect.feature_group_mixer import FeatureGroupMixer
from triage.component.architect.feature_group_creator import FeatureGroup

import pytest


@pytest.fixture
def english_numbers():
    return FeatureGroup(
        name="english_numbers",
        features_by_table={"one": ["two", "three"], "four": ["five", "six"]},
    )


@pytest.fixture
def letters():
    return FeatureGroup(
        name="letters", features_by_table={"a": ["b", "c"], "d": ["e", "f"]}
    )


@pytest.fixture
def german_numbers():
    return FeatureGroup(
        name="german_numbers",
        features_by_table={"eins": ["zwei", "drei"], "vier": ["funf", "sechs"]},
    )


def test_feature_group_mixer_leave_one_out(english_numbers, letters, german_numbers):
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(["leave-one-out"]).generate(feature_groups)
    expected = [
        dict(itertools.chain(letters.items(), german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), letters.items())),
    ]
    assert result == expected
    assert [g.names for g in result] == [
        ["letters", "german_numbers"],
        ["english_numbers", "german_numbers"],
        ["english_numbers", "letters"],
    ]


def test_feature_group_mixer_leave_one_in(english_numbers, letters, german_numbers):
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(["leave-one-in"]).generate(feature_groups)
    expected = [english_numbers, letters, german_numbers]
    assert result == expected
    assert [g.names for g in result] == [
        ["english_numbers"],
        ["letters"],
        ["german_numbers"],
    ]


def test_feature_group_mixer_all_combinations(english_numbers, letters,
                                              german_numbers):
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(['all-combinations']).generate(feature_groups)
    expected = [
        dict(itertools.chain(english_numbers.items())),
        dict(itertools.chain(letters.items())),
        dict(itertools.chain(german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), letters.items())),
        dict(itertools.chain(english_numbers.items(), german_numbers.items())),
        dict(itertools.chain(letters.items(), german_numbers.items())),
        dict(itertools.chain(english_numbers.items(), letters.items(),
                             german_numbers.items()))
    ]
    assert result == expected
    assert [g.names for g in result] == [
        ['english_numbers'],
        ['letters'],
        ['german_numbers'],
        ['english_numbers', 'letters'],
        ['english_numbers', 'german_numbers'],
        ['letters', 'german_numbers'],
        ['english_numbers', 'letters', 'german_numbers']
    ]

def test_feature_group_mixer_all(english_numbers, letters, german_numbers):
    feature_groups = [english_numbers, letters, german_numbers]

    result = FeatureGroupMixer(["all"]).generate(feature_groups)
    expected = [
        dict(
            itertools.chain(
                english_numbers.items(), letters.items(), german_numbers.items()
            )
        )
    ]
    assert result == expected
    assert result[0].names == ["english_numbers", "letters", "german_numbers"]
