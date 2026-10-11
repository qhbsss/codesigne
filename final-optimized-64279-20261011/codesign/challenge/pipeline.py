"""Instruction pipeline timing for the programmable Transformer challenge.

Functional execution remains in ``micro.py``. This module schedules the same
parsed instructions against issue, RF, execution, DMA, NoC and HBM resources.
The functional executor remains the authority on FP32 results. Timing uses
the same parsed instruction stream and explicit resource calendars.
"""

from dataclasses import dataclass
from math import prod
from tempfile import TemporaryFile

import numpy as np

from .hardware import Hardware
from .perf import PerfResult
from .power import EnergyEvent
from .resource_calendar import IntervalCalendar


@dataclass
class _MemoryTimes:
    written: np.ndarray
    read_until: np.ndarray
    backing: object | None = None

    @classmethod
    def sized(cls, words: int, sparse: bool = False):
        if sparse:
            # The ABI permits addresses near 2 GiB. A dense pair of int64
            # hazard arrays would occupy 8 GiB even if only a few words are
            # touched. Truncated temporary files are sparse and unlinked on
            # close; ndarray and strided-view operations retain their exact
            # element semantics while only visited pages consume storage.
            backing = TemporaryFile()
            backing.truncate(words * 16)
            written = np.memmap(backing, dtype=np.int64, mode="r+", shape=words, offset=0)
            read_until = np.memmap(
                backing, dtype=np.int64, mode="r+", shape=words, offset=words * 8
            )
            return cls(written, read_until, backing)
        return cls(np.zeros(words, np.int64), np.zeros(words, np.int64))

    def view(self, operand):
        offset = operand["offset"]
        shape = operand.get("shape", [operand["count"]])
        strides = operand.get("strides", [1])
        if (
            type(offset) is not int
            or offset < 0
            or type(operand["count"]) is not int
            or operand["count"] <= 0
            or not isinstance(shape, list)
            or not isinstance(strides, list)
            or len(shape) not in (1, 2)
            or len(shape) != len(strides)
            or any(type(x) is not int or x <= 0 for x in shape + strides)
            or prod(shape) != operand["count"]
        ):
            raise ValueError("Invalid timed memory view")
        span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
        if offset < 0 or offset + span > self.written.size:
            raise ValueError("Timed memory view out of bounds")
        if len(shape) == 1 and strides == [1]:
            return self.written[offset : offset + span], self.read_until[offset : offset + span]
        byte_strides = tuple(stride * 8 for stride in strides)
        return (
            np.lib.stride_tricks.as_strided(
                self.written[offset : offset + span], shape=shape, strides=byte_strides
            ),
            np.lib.stride_tricks.as_strided(
                self.read_until[offset : offset + span], shape=shape, strides=byte_strides
            ),
        )


@dataclass
class _GroupTimes:
    name: str
    sm: int
    words_per_lane: int
    next_issue: int
    last_finish: int
    rf: _MemoryTimes
    shared: _MemoryTimes


def _view(group: _GroupTimes, hbm: _MemoryTimes, operand):
    if operand["space"] == "HBM":
        if operand["wg"] is not None or operand["lane"] != 0:
            raise ValueError("HBM address/scope invalid")
        return hbm.view(operand)
    if operand["wg"] != group.name:
        raise ValueError("Cross-workgroup timed access")
    if operand["space"] == "RF":
        lane = operand["lane"]
        shape = operand.get("shape", [operand["count"]])
        strides = operand.get("strides", [1])
        span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
        if type(lane) is not int or not 0 <= lane < 16384 // group.words_per_lane:
            raise ValueError("RF lane invalid")
        if operand["offset"] + span > group.words_per_lane:
            raise ValueError("RF lane view out of bounds")
        adjusted = operand | {"offset": operand["offset"] + operand["lane"] * group.words_per_lane}
        return group.rf.view(adjusted)
    if operand["space"] == "SH":
        if operand["lane"] != 0:
            raise ValueError("Shared lane invalid")
        return group.shared.view(operand)
    raise ValueError("Invalid timed memory space")


def _max_write(group, hbm, operand):
    if "imm" in operand:
        return 0
    return int(_view(group, hbm, operand)[0].max())


def _max_destination(group, hbm, operand):
    written, read_until = _view(group, hbm, operand)
    return max(int(written.max()), int(read_until.max()))


def _mark(group, hbm, reads, writes, read_finish, write_finish):
    for operand in reads:
        if "imm" not in operand:
            read_until = _view(group, hbm, operand)[1]
            np.maximum(read_until, read_finish, out=read_until)
    for operand in writes:
        _view(group, hbm, operand)[0][...] = write_finish


def _reserve(free: list[IntervalCalendar], earliest: int, duration: int):
    if duration == 0:
        return earliest
    index = min(range(len(free)), key=lambda i: (free[i].first_free(earliest, duration), i))
    return free[index].reserve(earliest, duration)[1]


class _EnergyRuns:
    """Coalesce adjacent service slices of one instruction and resource."""

    def __init__(self, output: list[EnergyEvent]):
        self.output = output
        self.pending: dict[tuple[str, int], tuple[int, int, float, float]] = {}
        self.total = 0.0

    def add(self, key: tuple[str, int], start: int, finish: int, energy: float):
        if not energy:
            return
        self.total += energy
        rate = energy / (finish - start)
        old = self.pending.get(key)
        if old and old[1] == start and abs(old[3] - rate) < 1e-9:
            self.pending[key] = (old[0], finish, old[2] + energy, rate)
        else:
            if old:
                self.output.append(EnergyEvent(old[0], old[1], old[2]))
            self.pending[key] = (start, finish, energy, rate)

    def flush(self):
        self.output.extend(
            EnergyEvent(start, finish, energy) for start, finish, energy, _ in self.pending.values()
        )


def estimate_pipeline(
    hardware: Hardware,
    program,
    hbm_words: int,
    step_requirements: dict[int, list[tuple[int, int]]] | None = None,
    trace: dict[str, dict[str, int]] | None = None,
    energy_trace: list[EnergyEvent] | None = None,
    *,
    hbm_races_validated: bool = False,
) -> PerfResult:
    """Schedule the parsed program with the global event-time resource model."""
    from .pipeline_events import estimate_global_events

    return estimate_global_events(
        hardware,
        program,
        hbm_words,
        step_requirements,
        trace,
        energy_trace,
        hbm_races_validated=hbm_races_validated,
    )
