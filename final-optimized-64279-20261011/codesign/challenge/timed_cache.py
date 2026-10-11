"""Event-time model of the optional global read cache.

The caller owns HBM and network queues.  This class owns cache tag state and
one-request-per-cycle arbitration for each of four fixed cache banks.  Read
arrivals must not precede the last resolved lookup; a later call may not
schedule an earlier write completion.  An instruction scheduler with
overtaking requests must sort cache events before resolving tags.
"""

from collections import OrderedDict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable

from .resource_calendar import IntervalCalendar

HBM_BYTES = 2**31
LINE_BYTES = 64
WAYS = 4
BANKS = 4
HIT_LATENCY = 8


@dataclass
class TimedCacheTraffic:
    queries: int = 0
    hits: int = 0
    misses: int = 0
    bypass_reads: int = 0
    fills: int = 0
    invalidations: int = 0
    hbm_read_bytes: int = 0
    cache_read_bytes: int = 0
    cache_fill_bytes: int = 0


@dataclass(frozen=True)
class CacheAccess:
    lookup_cycle: int | None
    fill_cycle: int | None
    ready_cycle: int
    hit: bool
    hbm_requested: bool


@dataclass(frozen=True)
class PendingLookup:
    line: int
    cycle: int


class TimedGlobalCache:
    """Deterministic 64 B, four-way, four-bank global cache timing.

    ``miss_ready`` receives ``(aligned_line_address, earliest_hbm_cycle)`` and
    returns the cycle at which HBM has finished returning the complete line.
    The earliest cycle is eight cycles after the bank accepts the lookup.
    ``request_read`` returns ``(data_ready_cycle, hit, hbm_requested)``.  Its
    ready cycle is before the caller's NoC and destination storage service.

    Each miss independently calls ``miss_ready`` even while an identical line
    is pending.  Fills reserve a bank cycle and update LRU only at completion.
    Writes bypass this read cache; call ``schedule_write_invalidate`` with the
    HBM completion cycle for every touched line.

    Capacity zero sends each read directly to HBM. Such reads increment
    ``bypass_reads`` and HBM bytes, without counting a Cache query or miss.
    """

    def __init__(self, capacity_mib: int):
        if type(capacity_mib) is not int or capacity_mib not in (0, 1, 2, 4, 8, 16):
            raise ValueError("Invalid cache capacity")
        self.set_count = capacity_mib * 1024 * 1024 // (LINE_BYTES * WAYS)
        self.sets: dict[int, OrderedDict[int, None]] = {}
        self.traffic = TimedCacheTraffic()
        self._bank_busy = [IntervalCalendar() for _ in range(BANKS)]
        # Heap entries: (completion cycle, priority, order, kind, line, lookup).
        # A fill completes before an invalidation at the same cycle, so a write
        # ending at that cycle leaves the line invalid.
        self._events: list[tuple[int, int, int, str, int, int]] = []
        self._order = 0
        self._resolved_through = -1
        self._last_invalidation: dict[int, int] = {}
        self.last_access: CacheAccess | None = None

    def locate(self, address: int) -> tuple[int, int, int]:
        """Return set, bank and tag; cache-off returns line, -1, -1."""
        line = self._line(address)
        if not self.set_count:
            return line, -1, -1
        set_index = line % self.set_count
        return set_index, set_index % BANKS, line // self.set_count

    @staticmethod
    def _line(address: int) -> int:
        if type(address) is not int or not 0 <= address < HBM_BYTES:
            raise ValueError("HBM address outside 2 GiB")
        return address // LINE_BYTES

    @staticmethod
    def _cycle(cycle: int) -> int:
        if type(cycle) is not int or cycle < 0:
            raise ValueError("Invalid cache cycle")
        return cycle

    def _event(self, finish: int, priority: int, kind: str, line: int, lookup: int = -1):
        heappush(self._events, (finish, priority, self._order, kind, line, lookup))
        self._order += 1

    def _advance(self, cycle: int):
        if cycle < self._resolved_through:
            raise ValueError("Cache events must be resolved in chronological order")
        while self._events and self._events[0][0] <= cycle:
            finish, _, _, kind, line, lookup = heappop(self._events)
            set_index, _, tag = self.locate(line * LINE_BYTES)
            ways = self.sets.get(set_index)
            if kind == "invalidate":
                self._last_invalidation[line] = finish
                if ways is not None and tag in ways:
                    del ways[tag]
                    self.traffic.invalidations += 1
            else:
                self.traffic.fills += 1
                self.traffic.cache_fill_bytes += LINE_BYTES
                if self._last_invalidation.get(line, -1) <= lookup:
                    # A write completed after the miss lookup: its old HBM
                    # response still consumes the fill bank but cannot install
                    # a stale copy in the cache.
                    if ways is None:
                        ways = self.sets.setdefault(set_index, OrderedDict())
                    if tag in ways:
                        ways.move_to_end(tag)
                    else:
                        if len(ways) == WAYS:
                            ways.popitem(last=False)
                        ways[tag] = None
        self._resolved_through = cycle

    def _reserve_bank(self, bank: int, earliest: int) -> int:
        return self._bank_busy[bank].reserve(earliest, 1)[0]

    def request_read(
        self,
        address: int,
        arrival: int,
        miss_ready: Callable[[int, int], int],
    ) -> tuple[int, bool, bool]:
        """Schedule one line read and return its source ready cycle."""
        line = self._line(address)
        arrival = self._cycle(arrival)
        line_address = line * LINE_BYTES
        t = self.traffic
        if not self.set_count:
            ready = self._cycle(miss_ready(line_address, arrival))
            if ready < arrival:
                raise ValueError("HBM completed before cache-off request")
            t.bypass_reads += 1
            t.hbm_read_bytes += LINE_BYTES
            self.last_access = CacheAccess(None, None, ready, False, True)
            return ready, False, True

        pending = self.reserve_lookup(address, arrival)
        return self.resolve_lookup(pending, miss_ready)

    def reserve_lookup(self, address: int, arrival: int) -> PendingLookup:
        """Reserve a bank slot without resolving tags ahead of other events."""
        if not self.set_count:
            raise ValueError("Cache-off requests have no lookup")
        line = self._line(address)
        arrival = self._cycle(arrival)
        if arrival < self._resolved_through:
            raise ValueError("Cache arrivals must be chronological")
        _, bank, _ = self.locate(address)
        return PendingLookup(line, self._reserve_bank(bank, arrival))

    def resolve_lookup(
        self,
        pending: PendingLookup,
        miss_ready: Callable[[int, int], int],
    ) -> tuple[int, bool, bool]:
        """Resolve a reserved lookup when its cycle reaches the event queue."""
        if not self.set_count or not isinstance(pending, PendingLookup):
            raise ValueError("Invalid pending cache lookup")
        line, lookup = pending.line, pending.cycle
        line_address = line * LINE_BYTES
        t = self.traffic
        set_index, bank, tag = self.locate(line * LINE_BYTES)
        self._advance(lookup)
        t.queries += 1
        ways = self.sets.get(set_index)
        if ways is not None and tag in ways:
            ways.move_to_end(tag)
            t.hits += 1
            t.cache_read_bytes += LINE_BYTES
            self.last_access = CacheAccess(lookup, None, lookup + HIT_LATENCY, True, False)
            return lookup + HIT_LATENCY, True, False

        t.misses += 1
        t.hbm_read_bytes += LINE_BYTES
        earliest_hbm = lookup + HIT_LATENCY
        hbm_done = self._cycle(miss_ready(line_address, earliest_hbm))
        if hbm_done < earliest_hbm:
            raise ValueError("HBM completed before cache miss request")
        fill_begin = self._reserve_bank(bank, hbm_done)
        fill_done = fill_begin + 1
        self._event(fill_done, 0, "fill", line, lookup)
        self.last_access = CacheAccess(lookup, fill_begin, fill_done, False, True)
        return fill_done, False, True

    def schedule_write_invalidate(self, address: int, finish: int) -> None:
        """Invalidate one line when its bypassed HBM write completes."""
        line = self._line(address)
        finish = self._cycle(finish)
        if not self.set_count:
            return
        if finish < self._resolved_through:
            raise ValueError("Cannot schedule a retroactive cache invalidation")
        self._event(finish, 1, "invalidate", line)

    def drain(self, through: int) -> None:
        """Apply pending fills and invalidations through a known cycle."""
        self._advance(self._cycle(through))
