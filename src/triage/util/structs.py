"""Classes representing simple but deep data structures that we reuse throughout
Triage code and want to display more intelligently in log files
"""


class TruncatedRepresentationList(list):
    def __repr__(self):
        return f"[{self[0]} ... {self[-1]} (Total: {len(self)})]"


class AsOfTimeList(TruncatedRepresentationList):
    pass


class FeatureNameList(TruncatedRepresentationList):
    def check(self, value):
        if not isinstance(value, str):
            raise TypeError("A FeatureNameList represents a list of feature names, and therefore"
                            f"each item must be a string, not: {value!r}")
        return value

    def insert(self, i, v):
        raise ValueError("in insert!")
        self.check(v)
        super().insert(i, v)

    def __setitem__(self, i, v):
        raise ValueError("in setitem!")
        self.check(v)
        super().__setitem__(i, v)

    def append(self, item):
        raise ValueError("in append!")
        self.check(item)
        super().append(item)

    def extend(self, t):
        raise ValueError("in extend")
        return super().extend([ self.check(v) for v in t ])

    def __add__(self, t): # This is for something like `CheckedList(validator, [1, 2, 3]) + list([4, 5, 6])`...
        raise ValueError("in add")
        return super().__add__([ self.check(v) for v in t ])

    def __iadd__(self, t): # This is for something like `l = CheckedList(validator); l += [1, 2, 3]`
        raise ValueError("in iadd")
        return super().__iadd__([ self.check(v) for v in t ])
