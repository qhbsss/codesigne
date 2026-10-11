"""Deterministic global v0.4 cache tags and traffic accounting.

Timing arbitration belongs to the event scheduler. This module handles one line
request at a time in scheduler order and deliberately does not merge pending misses.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheTraffic:
    queries: int = 0
    hits: int = 0
    misses: int = 0
    fills: int = 0
    invalidations: int = 0
    hbm_read_bytes: int = 0
    cache_read_bytes: int = 0
    cache_fill_bytes: int = 0


class GlobalCache:
    def __init__(self, capacity_mib: int):
        if type(capacity_mib) is not int or capacity_mib not in (0, 1, 2, 4, 8, 16):
            raise ValueError("Invalid cache capacity")
        self.set_count = capacity_mib * 1024 * 1024 // (64 * 4)
        self.sets: dict[int, OrderedDict[int, None]] = {}
        self.traffic = CacheTraffic()

    def locate(self, address: int) -> tuple[int, int, int]:
        if type(address) is not int or not 0 <= address < 2**31:
            raise ValueError("HBM address outside 2 GiB")
        line = address // 64
        if not self.set_count:
            return line, -1, -1
        set_index = line % self.set_count
        return set_index, set_index % 4, line // self.set_count

    def read_line(self, address: int) -> bool:
        """Return hit; account source bytes for one complete aligned line."""
        set_index, _, tag = self.locate(address)
        t = self.traffic
        if not self.set_count:
            t.misses += 1
            t.hbm_read_bytes += 64
            return False
        t.queries += 1
        ways = self.sets.setdefault(set_index, OrderedDict())
        if tag in ways:
            t.hits += 1
            t.cache_read_bytes += 64
            ways.move_to_end(tag)
            return True
        t.misses += 1
        t.hbm_read_bytes += 64
        t.cache_fill_bytes += 64
        t.fills += 1
        if len(ways) == 4:
            ways.popitem(last=False)
        ways[tag] = None
        return False

    def write(self, address: int, length: int) -> None:
        if type(length) is not int or length <= 0 or address + length > 2**31:
            raise ValueError("HBM write outside 2 GiB")
        self.locate(address)
        if not self.set_count:
            return
        for line in range(address // 64, (address + length - 1) // 64 + 1):
            set_index, _, tag = self.locate(line * 64)
            ways = self.sets.get(set_index)
            if ways is not None and tag in ways:
                del ways[tag]
                self.traffic.invalidations += 1
