from collate import collate

def assert_contains(haystack, needle):
    for h in haystack:
        if needle in h: return
    assert False

def test_multicompare_lists():
    d = collate.MultiCompare('col','=',['a','b','c'],[]).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")
    assert_contains(map(str.lower, d.values()), "col is null")

    d = collate.MultiCompare('col','>',[1,2,3],[]).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col > 1")
    assert_contains(d.values(), "col > 2")
    assert_contains(d.values(), "col > 3")
    assert_contains(map(str.lower, d.values()), "col is null")

    d = collate.MultiCompare('col','=',['a','b','c'], [], include_null=False).quantities
    assert len(d) == 3
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")

    d = collate.MultiCompare('really_long_column_name','=',
        ['really long string value that is similar to others',
         'really long string value that is like others',
         'really long string value that is quite alike to others',
         'really long string value that is also like everything else'], [], maxlen=32).quantities
    assert len(d) == 5
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert all(len(k) <= 32 for k in d.keys())
    assert_contains(d.values(), "really_long_column_name = 'really long string value that is similar to others'")
    assert_contains(d.values(), "really_long_column_name = 'really long string value that is like others'")
    assert_contains(d.values(), "really_long_column_name = 'really long string value that is quite alike to others'")
    assert_contains(d.values(), "really_long_column_name = 'really long string value that is also like everything else'")
    assert_contains(map(str.lower, d.values()), "really_long_column_name is null")

def test_multicompare_dicts():
    d = collate.MultiCompare('col','=',{'vala': 'a','valb': 'b','valc': 'c'}, []).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col = 'a'")
    assert_contains(d.values(), "col = 'b'")
    assert_contains(d.values(), "col = 'c'")
    assert_contains(d.keys(), 'vala')
    assert_contains(d.keys(), 'valb')
    assert_contains(d.keys(), 'valc')
    assert_contains(map(str.lower, d.keys()), 'null')
    assert_contains(map(str.lower, d.values()), "col is null")

    d = collate.MultiCompare('col','<',{'val1': 1,'val2': 2,'val3': 3}, []).quantities
    assert len(d) == 4
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert_contains(d.values(), "col < 1")
    assert_contains(d.values(), "col < 2")
    assert_contains(d.values(), "col < 3")
    assert_contains(d.keys(), 'val1')
    assert_contains(d.keys(), 'val2')
    assert_contains(d.keys(), 'val3')
    assert_contains(map(str.lower, d.keys()), 'null')
    assert_contains(map(str.lower, d.values()), "col is null")

    d = collate.MultiCompare('long_column_name','=',
        {'really long string key that is similar to others': 'really long string value that is similar to others',
         'really long string key that is like others': 'really long string value that is like others',
         'different key': 'really long string value that is quite alike to others',
         'ni': 'really long string value that is also like everything else'}, [], maxlen=32).quantities
    assert len(d) == 5
    assert len(set(d.values())) == len(d)
    assert len(set(d.keys())) == len(d)
    assert all(len(k) <= 32 for k in d.keys())
    assert_contains(d.keys(), 'differ')
    assert_contains(d.values(), "long_column_name = 'really long string value that is similar to others'")
    assert_contains(d.values(), "long_column_name = 'really long string value that is like others'")
    assert_contains(d.values(), "long_column_name = 'really long string value that is quite alike to others'")
    assert_contains(d.values(), "long_column_name = 'really long string value that is also like everything else'")
    assert_contains(map(str.lower, d.values()), "long_column_name is null")
