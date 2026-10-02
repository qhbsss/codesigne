"""Exact integer-cycle sliding power window for uniformly spread event energy."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .hardware import cost_model

_WINDOW_BATCH_EDGES = 1_000_000


@dataclass(frozen=True, slots=True)
class EnergyEvent:
    start: int
    finish: int
    dynamic_pj: float


def max_window_power_w(
    area_mm2: float, events: list[EnergyEvent], window_cycles: int | None = None
) -> float:
    return power_profile_w(area_mm2, events, window_cycles)[0]


def power_profile_w(
    area_mm2: float, events: list[EnergyEvent], window_cycles: int | None = None
) -> tuple[float, float]:
    """Max average W over every integer-aligned window of fixed length.

    An event's dynamic energy is uniform on [start, finish). The full window
    may include an idle tail; static power is charged throughout the window.
    Also return the highest single-cycle power as an upper bound on any
    longer-window average.
    """
    cost = cost_model()
    window = cost["power_window_cycles"] if window_cycles is None else window_cycles
    if not isfinite(area_mm2) or area_mm2 < 0 or type(window) is not int or window <= 0:
        raise ValueError("Invalid area/window")
    for event in events:
        if (
            type(event.start) is not int
            or type(event.finish) is not int
            or event.start < 0
            or event.finish <= event.start
            or not isfinite(event.dynamic_pj)
            or event.dynamic_pj < 0
        ):
            raise ValueError("Invalid energy event")
    if not events:
        static = area_mm2 * cost["static_w_per_mm2"]
        return static, static
    edges = np.fromiter(
        (edge for event in events for edge in (event.start, event.finish)),
        dtype=np.int64,
        count=2 * len(events),
    )
    changes = np.fromiter(
        (
            signed * event.dynamic_pj / (event.finish - event.start)
            for event in events
            for signed in (1, -1)
        ),
        dtype=np.float64,
        count=2 * len(events),
    )
    order = np.argsort(edges, kind="stable")
    edges, changes = edges[order], changes[order]
    del order
    unique_edges, first = np.unique(edges, return_index=True)
    changes = np.add.reduceat(changes, first)
    del edges, first
    rates = np.cumsum(changes)
    del changes
    rates[-1] = 0.0
    instantaneous_pj_per_cycle = float(np.max(rates))
    widths = np.diff(unique_edges)
    prefix = np.empty(len(unique_edges), dtype=np.float64)
    prefix[0] = 0
    prefix[1:] = np.cumsum(widths * rates[:-1])
    del widths

    def integral_at(points):
        index = np.searchsorted(unique_edges, points, side="right") - 1
        np.clip(index, 0, len(unique_edges) - 1, out=index)
        return prefix[index] + (points - unique_edges[index]) * rates[index]

    # All candidate starts are unique event edges or an edge minus the window.
    # Process them in bounded batches so full-size power runs do not hold
    # several additional arrays proportional to the event count at once.
    maximum_pj = 0.0
    for offset in range(0, len(unique_edges), _WINDOW_BATCH_EDGES):
        batch = unique_edges[offset : offset + _WINDOW_BATCH_EDGES]
        for candidates in (batch, np.maximum(0, batch - window)):
            gain = integral_at(candidates + window) - integral_at(candidates)
            maximum_pj = max(maximum_pj, float(np.max(gain)))
    cycle_seconds = 1 / cost["clock_hz"]
    dynamic_w = maximum_pj * 1e-12 / (window * cycle_seconds)
    static = area_mm2 * cost["static_w_per_mm2"]
    instantaneous_w = instantaneous_pj_per_cycle * 1e-12 / cycle_seconds
    return static + dynamic_w, static + instantaneous_w
