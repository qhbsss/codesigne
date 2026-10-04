"""Resource calendars that permit reservations before already booked future work.

IntervalCalendar models an exclusive unit. ByteCalendar models a byte-per-cycle
link and allows unrelated transfers to share a cycle's remaining bandwidth.
Neither class advances a global cursor, so a later call may fill an earlier gap.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable


def _nonnegative(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _priority(key: int) -> int:
    """Stable, well-mixed treap priority without a mutable random seed."""
    mask = (1 << 64) - 1
    value = (key + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


@dataclass(slots=True)
class _Interval:
    start: int
    end: int
    priority: int
    left: "_Interval | None" = None
    right: "_Interval | None" = None
    used: int = 0


def _rotate_left(root: _Interval) -> _Interval:
    child = root.right
    assert child is not None
    root.right = child.left
    child.left = root
    return child


def _rotate_right(root: _Interval) -> _Interval:
    child = root.left
    assert child is not None
    root.left = child.right
    child.right = root
    return child


def _insert(root: _Interval | None, item: _Interval) -> _Interval:
    if root is None:
        return item
    if item.start < root.start:
        root.left = _insert(root.left, item)
        if root.left.priority < root.priority:
            return _rotate_right(root)
    else:
        root.right = _insert(root.right, item)
        if root.right.priority < root.priority:
            return _rotate_left(root)
    return root


def _merge(left: _Interval | None, right: _Interval | None) -> _Interval | None:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        left.right = _merge(left.right, right)
        return left
    right.left = _merge(left, right.left)
    return right


def _erase(root: _Interval | None, start: int) -> _Interval | None:
    assert root is not None
    if start == root.start:
        return _merge(root.left, root.right)
    if start < root.start:
        root.left = _erase(root.left, start)
    else:
        root.right = _erase(root.right, start)
    return root


def _occupied_cycles(root: _Interval | None) -> int:
    total = 0
    pending = [root] if root is not None else []
    while pending:
        item = pending.pop()
        total += item.end - item.start
        if item.left is not None:
            pending.append(item.left)
        if item.right is not None:
            pending.append(item.right)
    return total


class IntervalCalendar:
    """Reserve nonoverlapping half-open intervals on one exclusive resource.

    Storage is a deterministic treap of merged busy intervals. A search and
    insertion are expected O(log n), including when requests arrive out of
    start-time order. Consecutive reservations merge into a single interval.
    """

    def __init__(self):
        self._root: _Interval | None = None
        self.interval_count = 0

    def busy_cycles(self) -> int:
        """Return the union of occupied cycles on this exclusive resource."""
        return _occupied_cycles(self._root)

    def iter_intervals(self):
        """Yield merged ``(start, end)`` reservations in cycle order."""
        pending = []
        node = self._root
        while pending or node is not None:
            while node is not None:
                pending.append(node)
                node = node.left
            node = pending.pop()
            yield node.start, node.end
            node = node.right

    def _first_ending_after(self, cycle: int) -> _Interval | None:
        node = self._root
        candidate = None
        while node is not None:
            if node.end <= cycle:
                node = node.right
            else:
                candidate = node
                node = node.left
        return candidate

    def _neighbors(self, start: int) -> tuple[_Interval | None, _Interval | None]:
        previous = following = None
        node = self._root
        while node is not None:
            if node.start < start:
                previous = node
                node = node.right
            else:
                following = node
                node = node.left
        return previous, following

    def reserve(self, earliest: int, duration: int) -> tuple[int, int]:
        """Book the first free interval of ``duration`` cycles at/after earliest."""
        start = self.first_free(earliest, duration)
        end = start + duration
        previous, following = self._neighbors(start)
        if previous is not None and previous.end == start:
            previous.end = end
            if following is not None and following.start == end:
                previous.end = following.end
                self._root = _erase(self._root, following.start)
                self.interval_count -= 1
        elif following is not None and following.start == end:
            # start remains between the same predecessor and successor, so
            # moving the successor's key left preserves the search-tree order.
            following.start = start
        else:
            self._root = _insert(self._root, _Interval(start, end, _priority(start)))
            self.interval_count += 1
        return start, end

    def first_free(self, earliest: int, duration: int) -> int:
        """Find the first available start without making a reservation."""
        start = _nonnegative(earliest, "earliest")
        duration = _positive(duration, "duration")
        while (conflict := self._first_ending_after(start)) is not None:
            if conflict.start >= start + duration:
                break
            start = conflict.end
        return start


class ByteCalendar:
    """A byte-per-cycle link stored as coalesced runs of equal occupancy.

    Long sequences at either full or partial utilization need one tree node.
    Updating a cycle within a run splits it, then joins equal neighboring
    runs again. This also permits reservations before already booked future
    traffic without keeping a dictionary entry for every service cycle.
    """

    def __init__(self, capacity: int):
        self.capacity = _positive(capacity, "capacity")
        self._root: _Interval | None = None
        self._tail: _Interval | None = None
        self.interval_count = 0

    def _rightmost(self) -> _Interval | None:
        node = self._root
        if node is not None:
            while node.right is not None:
                node = node.right
        return node

    def busy_cycles(self) -> int:
        """Count cycles with any booked bytes, independent of utilization."""
        return _occupied_cycles(self._root)

    def iter_runs(self):
        """Yield ``(start, end, bytes_per_cycle)`` occupied runs in cycle order."""
        pending = []
        node = self._root
        while pending or node is not None:
            while node is not None:
                pending.append(node)
                node = node.left
            node = pending.pop()
            yield node.start, node.end, node.used
            node = node.right

    def _containing(self, cycle: int) -> _Interval | None:
        tail = self._tail
        if tail is not None and cycle >= tail.start:
            return tail if cycle < tail.end else None
        node = self._root
        while node is not None:
            if cycle < node.start:
                node = node.left
            elif cycle >= node.end:
                node = node.right
            else:
                return node
        return None

    def _neighbors(self, start: int) -> tuple[_Interval | None, _Interval | None]:
        previous = following = None
        node = self._root
        while node is not None:
            if node.start < start:
                previous = node
                node = node.right
            else:
                following = node
                node = node.left
        return previous, following

    def _insert_run(self, start: int, end: int, used: int) -> _Interval:
        item = _Interval(start, end, _priority(start), used=used)
        self._root = _insert(self._root, item)
        self.interval_count += 1
        return item

    def _erase_run(self, start: int) -> None:
        self._root = _erase(self._root, start)
        self.interval_count -= 1

    def _add(self, cycle: int, amount: int) -> None:
        tail = self._tail
        if (tail is None or cycle >= tail.end) and amount > self.capacity:
            raise ValueError("Byte calendar capacity exceeded")
        if tail is not None and cycle == tail.end and tail.used == amount:
            tail.end += 1
            return
        if tail is None or cycle >= tail.end:
            self._tail = self._insert_run(cycle, cycle + 1, amount)
            return
        old = self._containing(cycle)
        old_used = old.used if old is not None else 0
        new_used = old_used + amount
        if new_used > self.capacity:
            raise ValueError("Byte calendar capacity exceeded")

        if old is None:
            current = self._insert_run(cycle, cycle + 1, new_used)
        elif old.start == cycle and old.end == cycle + 1:
            old.used = new_used
            current = old
        else:
            start, end = old.start, old.end
            self._erase_run(start)
            if start < cycle:
                self._insert_run(start, cycle, old_used)
            current = self._insert_run(cycle, cycle + 1, new_used)
            if cycle + 1 < end:
                self._insert_run(cycle + 1, end, old_used)

        previous, _ = self._neighbors(current.start)
        if previous is not None and previous.end == current.start and previous.used == new_used:
            previous.end = current.end
            self._erase_run(current.start)
            current = previous
        _, following = self._neighbors(current.end)
        if following is not None and following.start == current.end and following.used == new_used:
            current.end = following.end
            self._erase_run(following.start)
        self._tail = self._rightmost()

    def used_at(self, cycle: int) -> int:
        item = self._containing(_nonnegative(cycle, "cycle"))
        return item.used if item is not None else 0

    def _next_available(self, cycle: int) -> int:
        item = self._containing(cycle)
        if item is not None and item.used == self.capacity:
            return item.end
        return cycle

    def reserve(
        self,
        earliest: int,
        amount: int,
        on_service: Callable[[int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Book bytes greedily in available cycles; return first and final boundary."""
        return reserve_linked_bytes((self,), earliest, amount, on_service=on_service)


def reserve_linked_bytes(
    calendars: Sequence[ByteCalendar],
    earliest: int,
    amount: int,
    on_service: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Book the same bytes simultaneously on every link in ``calendars``.

    For example, an HBM-to-SM transfer can reserve 64 B on both the global NoC
    and the destination SM link. The links may already carry other transfers
    in a cycle; only their remaining common capacity is used.
    """
    cycle = _nonnegative(earliest, "earliest")
    remaining = _positive(amount, "amount")
    if not calendars or len({id(item) for item in calendars}) != len(calendars):
        raise ValueError("Provide distinct byte calendars")
    first = None
    while remaining:
        cycle = max(item._next_available(cycle) for item in calendars)
        available = min(item.capacity - item.used_at(cycle) for item in calendars)
        if available:
            take = min(remaining, available)
            for item in calendars:
                item._add(cycle, take)
            if first is None:
                first = cycle
            if on_service is not None:
                on_service(cycle, take)
            remaining -= take
        cycle += 1
    assert first is not None
    return first, cycle
