from triage.component.collate import Categorical, Compare


def assert_contains(haystack, needle):
    for h in haystack:
        if type(h) is tuple and needle in h[0]:
            return
        elif needle in h:
            return
    assert False


def test_compare_lists():
    d = Compare("col", "=", ["a", "b", "c"], [], {}, include_null=True).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")
    assert_contains(map(lambda x: x[0].lower(), d.values()), "col is null")

    d = Compare("col", ">", [1, 2, 3], [], {}).quantities
    assert len(d) == 3
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col > 1")
    assert_contains(d.values(), "col > 2")
    assert_contains(d.values(), "col > 3")

    d = Compare("col", "=", ["a", "b", "c"], [], {}, include_null=False).quantities
    assert len(d) == 3
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")

    d = Compare(
        "really_long_column_name",
        "=",
        [
            "really long string value that is similar to others",
            "really long string value that is like others",
            "really long string value that is quite alike to others",
            "really long string value that is also like everything else",
        ],
        [],
        {},
        maxlen=32,
    ).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert all(len(k) <= 32 for k in d.keys())
    assert_contains(
        d.values(),
        "really_long_column_name = 'really long string value that is similar to others'",
    )
    assert_contains(
        d.values(),
        "really_long_column_name = 'really long string value that is like others'",
    )
    assert_contains(
        d.values(),
        "really_long_column_name = 'really long string value that is quite alike to others'",
    )
    assert_contains(
        d.values(),
        "really_long_column_name = 'really long string value that is also like everything else'",
    )


def test_compare_override_quoting():
    d = Compare(
        "col",
        "@>",
        {"one": "array['one'::varchar]", "two": "array['two'::varchar]"},
        [],
        {},
        quote_choices=False,
    ).quantities
    assert len(d) == 2
    assert_contains(d.values(), "col @> array['one'::varchar]")
    assert_contains(d.values(), "col @> array['two'::varchar]")


def test_compare_dicts():
    d = Compare(
        "col", "=", {"vala": "a", "valb": "b", "valc": "c"}, [], {}, include_null=True
    ).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")
    assert_contains(d.keys(), "vala")
    assert_contains(d.keys(), "valb")
    assert_contains(d.keys(), "valc")
    assert_contains(map(str.lower, d.keys()), "null")
    assert_contains(map(lambda x: x[0].lower(), d.values()), "col is null")

    d = Compare(
        "col", "<", {"val1": 1, "val2": 2, "val3": 3}, [], {}, include_null="missing"
    ).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col < 1")
    assert_contains(d.values(), "col < 2")
    assert_contains(d.values(), "col < 3")
    assert_contains(map(lambda x: x[0].lower(), d.values()), "null")
    assert_contains(d.keys(), "val1")
    assert_contains(d.keys(), "val2")
    assert_contains(d.keys(), "val3")
    assert_contains(d.keys(), "missing")

    d = Compare(
        "long_column_name",
        "=",
        {
            "really long string key that is similar to others":
                "really long string value that is similar to others",
            "really long string key that is like others":
                "really long string value that is like others",
            "different key": "really long string value that is quite alike to others",
            "ni": "really long string value that is also like everything else",
        },
        [],
        {},
        maxlen=32,
    ).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert all(len(k) <= 32 for k in d.keys())
    assert_contains(d.keys(), "differ")
    assert_contains(
        d.values(),
        "long_column_name = 'really long string value that is similar to others'",
    )
    assert_contains(
        d.values(), "long_column_name = 'really long string value that is like others'"
    )
    assert_contains(
        d.values(),
        "long_column_name = 'really long string value that is quite alike to others'",
    )
    assert_contains(
        d.values(),
        "long_column_name = 'really long string value that is also like everything else'",
    )


def test_categorical_same_as_compare():
    d1 = Categorical("col", {"vala": "a", "valb": "b", "valc": "c"}, [], {}).quantities
    d2 = Compare("col", "=", {"vala": "a", "valb": "b", "valc": "c"}, [], {}).quantities
    assert sorted(d1.values()) == sorted(d2.values())
    d3 = Categorical(
        "col", {"vala": "a", "valb": "b", "valc": "c"}, [], {}, op_in_name=True
    ).quantities
    assert d2 == d3


def test_categorical_nones():
    d1 = Categorical(
        "col", {"vala": "a", "valb": "b", "valc": "c", "_NULL": None}, [], {}
    ).quantities
    d2 = Compare(
        "col",
        "=",
        {"vala": "a", "valb": "b", "valc": "c"},
        [],
        {},
        op_in_name=False,
        include_null=True,
    ).quantities
    assert d1 == d2
    d3 = Categorical("col", ["a", "b", "c", None], [], {}).quantities
    assert sorted(d1.values()) == sorted(d3.values())
