from collections import namedtuple
import pytorch_lightning as pl


class EEGDataset(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()


EEGBatch = namedtuple(
    "EEGBatch",
    (
        "data",
        "label",
        "id",
        "sample_id",
        "patient",
        "seizure_start",
        "seizure_end",
        "done",
        "sampling_rate",
    ),
)
PatientData = namedtuple("PatientData", ("id", "seizures", "channels"))


def consolidate(intervals):
    sorted_intervals = sorted(intervals)

    if not sorted_intervals:
        return

    low, high = sorted_intervals[0]

    for iv in sorted_intervals[1:]:
        if iv[0] <= high:
            high = max(high, iv[1])
        else:
            yield [low, high]
            low, high = iv

    yield [low, high]


def intervals_intersection(a, b):
    ranges = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right < b_right:
            i += 1
        else:
            j += 1

        if a_right >= b_left and b_right >= a_left:
            end_pts = sorted([a_left, a_right, b_left, b_right])
            middle = [end_pts[1], end_pts[2]]
            ranges.append(middle)

    ri = 0
    while ri < len(ranges) - 1:
        if ranges[ri][1] == ranges[ri + 1][0]:
            ranges[ri : ri + 2] = [[ranges[ri][0], ranges[ri + 1][1]]]

        ri += 1

    return ranges
