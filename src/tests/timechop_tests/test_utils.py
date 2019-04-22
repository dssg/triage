from triage.component.timechop.utils import convert_to_list


def test_convert_to_list():
    tests = [
        {"val": "1 day", "expected_result": ["1 day"]},
        {"val": ["1 day"], "expected_result": ["1 day"]},
        {"val": 1, "expected_result": [1]},
    ]
    for test in tests:
        assert convert_to_list(test["val"]) == test["expected_result"]
