"""Compact, exact interval integrals for the challenge resource report.

The scheduler already retains resource calendars and energy service runs. This
module projects those intervals onto at most 128 cycle bins at report time;
it does not retain a per-cycle or per-instruction trace.
"""

from math import isclose

from .power import EnergyEvent
from .resource_calendar import ByteCalendar, IntervalCalendar


class _Bins:
    """Integrate a constant rate over half-open cycle intervals."""

    def __init__(self, edges: list[int]):
        self.edges = edges
        self.width = edges[1] - edges[0]
        self.values = [0] * (len(edges) - 1)
        self.difference = [0] * len(edges)

    def add(self, start: int, end: int, rate: int | float) -> None:
        if not 0 <= start < end <= self.edges[-1]:
            raise ValueError("Timeline interval exceeds measured execution")
        first = start // self.width
        last = (end - 1) // self.width
        if first == last:
            self.values[first] += (end - start) * rate
            return
        self.values[first] += (self.edges[first + 1] - start) * rate
        self.values[last] += (end - self.edges[last]) * rate
        self.difference[first + 1] += rate
        self.difference[last] -= rate

    def finish(self) -> list[int | float]:
        active = 0
        for index, (start, end) in enumerate(zip(self.edges, self.edges[1:])):
            active += self.difference[index]
            self.values[index] += active * (end - start)
        return self.values


class ResourceTimeline:
    """Build JSON-ready resource and power bins from finalized calendars."""

    def __init__(self, cycles: int, max_bins: int = 128):
        if type(cycles) is not int or cycles <= 0 or type(max_bins) is not int or max_bins <= 0:
            raise ValueError("Invalid timeline size")
        width = max(1, (cycles + max_bins - 1) // max_bins)
        self.edges = list(range(0, cycles, width)) + [cycles]
        self._busy: dict[str, dict[str, tuple[_Bins, int]]] = {}
        self._bytes: dict[str, dict[str, tuple[_Bins, int]]] = {}

    def _scope(self, sm: int | None) -> str:
        return "shared" if sm is None else str(sm)

    def add_exclusive(
        self, sm: int | None, resource: str, calendars: list[IntervalCalendar]
    ) -> None:
        if not calendars:
            raise ValueError("Timeline resource has no entities")
        bins = _Bins(self.edges)
        for calendar in calendars:
            for start, end in calendar.iter_intervals():
                bins.add(start, end, 1)
        self._busy.setdefault(self._scope(sm), {})[resource] = (bins, len(calendars))

    def add_bandwidth(self, sm: int | None, resource: str, calendar: ByteCalendar) -> None:
        busy, transferred = _Bins(self.edges), _Bins(self.edges)
        for start, end, used in calendar.iter_runs():
            busy.add(start, end, 1)
            transferred.add(start, end, used)
        scope = self._scope(sm)
        self._busy.setdefault(scope, {})[resource] = (busy, 1)
        self._bytes.setdefault(scope, {})[resource] = (transferred, calendar.capacity)

    def finish(
        self,
        events: list[EnergyEvent],
        dynamic_energy_pj: float,
        static_power_w: float,
        clock_hz: int,
        power_window_cycles: int,
        exact_peak_window_w: float,
    ) -> dict:
        widths = [end - start for start, end in zip(self.edges, self.edges[1:])]
        scopes = {}
        for scope, resources in self._busy.items():
            busy = {resource: bins.finish() for resource, (bins, _) in resources.items()}
            utilization = {
                resource: [
                    used / (width * resources[resource][1]) for used, width in zip(values, widths)
                ]
                for resource, values in busy.items()
            }
            byte_resources = self._bytes.get(scope, {})
            transferred = {
                resource: bins.finish() for resource, (bins, _) in byte_resources.items()
            }
            bandwidth_utilization = {
                resource: [
                    used / (width * byte_resources[resource][1])
                    for used, width in zip(values, widths)
                ]
                for resource, values in transferred.items()
            }
            scopes[scope] = {
                "busy_cycles": busy,
                "utilization": utilization,
                "bytes": transferred,
                "bandwidth_utilization": bandwidth_utilization,
            }
        energy = _Bins(self.edges)
        for event in events:
            energy.add(
                event.start,
                event.finish,
                event.dynamic_pj / (event.finish - event.start),
            )
        energy_values = energy.finish()
        if not isclose(sum(energy_values), dynamic_energy_pj, rel_tol=1e-9, abs_tol=1e-5):
            raise AssertionError("Timeline energy does not conserve the service trace")
        return {
            "version": "challenge-resource-timeline-v1",
            "edges_cycles": self.edges,
            "sms": {scope: values for scope, values in scopes.items() if scope != "shared"},
            "shared": scopes.get("shared", {}),
            "power": {
                "dynamic_energy_pj": energy_values,
                "average_w": [
                    static_power_w + amount * clock_hz * 1e-12 / width
                    for amount, width in zip(energy_values, widths)
                ],
                "exact_peak_window_w": exact_peak_window_w,
                "window_cycles": power_window_cycles,
            },
        }
