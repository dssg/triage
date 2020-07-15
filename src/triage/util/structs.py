"""Classes representing simple but deep data structures that we reuse throughout
Triage code and want to display more intelligently in log files
"""


class TruncatedRepresentationList(list):
    def __repr__(self):
        total = len(self)
        if total != 1:
            return f"[{self[0]} ... {self[-1]} (Total: {total})]"
        else:
            return f"[{self[0]}] (Total: {total})"


class AsOfTimeList(TruncatedRepresentationList):
    pass


class FeatureNameList(TruncatedRepresentationList):
    pass
