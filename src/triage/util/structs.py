"""Classes representing simple but deep data structures that we reuse throughout
Triage code and want to display more intelligently in log files
"""


class TruncatedRepresentationList(list):
    def __repr__(self):
        return f"[{self[0]} ... {self[-1]} (Total: {len(self)})]"


class AsOfTimeList(TruncatedRepresentationList):
    pass


class FeatureNameList(TruncatedRepresentationList):
    pass
