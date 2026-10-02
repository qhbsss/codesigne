"""Program order and event-time primitives for the challenge pipeline.

The source text is one deterministic description of several workgroups. Each
workgroup keeps its own instruction order. A shared barrier token appears in
every participating stream, so independent groups may issue around a stalled
peer while the barrier itself still joins them at one point in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil
from typing import Callable

from .address_views import overlap as _address_overlap
from .address_views import positions as _positions
from .hardware import Hardware, cost_model
from .isa import Instruction, sync_targets
from .perf import PerfResult, _lines
from .power import EnergyEvent, power_profile_w
from .resource_calendar import ByteCalendar, IntervalCalendar, reserve_linked_bytes
from .service import mma_service, reduction_cycles, sfu_cycles, vector_cycles
from .timed_cache import CacheAccess, TimedGlobalCache


@dataclass(eq=False)
class GroupTape:
    name: str
    sm: int
    shared_bytes: int
    commands: list[tuple[int, Instruction] | None] = field(default_factory=list)
    cursor: int = 0
    predecessor: GroupTape | None = None
    start_after_commit: int | None = None
    control: bool = False
    end_cycle: int | None = None

    def next_command(self) -> tuple[int, Instruction] | None:
        return self.commands[self.cursor] if self.cursor < len(self.commands) else None


@dataclass(frozen=True)
class ProgramQueues:
    groups: tuple[GroupTape, ...]
    sync_participants: dict[int, tuple[GroupTape, ...]]
    commit_closed: dict[int, tuple[GroupTape, ...]]
    instruction_count: int
    waited_events: frozenset[str]


def group_program(program) -> ProgramQueues:
    """Assign every parsed instruction to its scoped workgroup stream.

    ``sync_participants`` is keyed by the object id of one shared WAIT,
    BARRIER or STEP.COMMIT instruction. The same instance is appended to each
    selected group. A later WG.BEGIN may reuse a textual group name after the
    prior WG.END; it becomes a distinct tape with fresh local storage.
    """
    active: dict[str, GroupTape] = {}
    closed: dict[str, GroupTape] = {}
    all_groups: list[GroupTape] = []
    sync_participants: dict[int, tuple[GroupTape, ...]] = {}
    commit_closed: dict[int, tuple[GroupTape, ...]] = {}
    last_commit: int | None = None
    seen_events: set[str] = set()
    waited_events: set[str] = set()
    next_step = 0
    count = 0
    for pc, ins in enumerate(program):
        if not isinstance(ins, Instruction):
            raise ValueError("Expected parsed challenge instruction")
        count += 1
        op, args = ins.op, ins.args
        if op == "WG.BEGIN":
            name = args["wg"]
            if name in active:
                raise ValueError("Duplicate active workgroup")
            tape = GroupTape(
                name,
                args["sm"],
                args["shared_bytes"],
                predecessor=closed.get(name),
                start_after_commit=last_commit,
            )
            tape.commands.append((pc, ins))
            active[name] = tape
            all_groups.append(tape)
        elif op == "WG.END":
            tape = active.pop(args["wg"], None)
            if tape is None:
                raise ValueError("WG.END for inactive workgroup")
            tape.commands.append((pc, ins))
            closed[tape.name] = tape
        elif op in ("WAIT", "BARRIER", "STEP.COMMIT"):
            if op == "STEP.COMMIT":
                if type(args.get("step")) is not int or args["step"] != next_step:
                    raise ValueError("Invalid step commitment")
                targets = tuple(active.values())
                commit_closed[id(ins)] = tuple(
                    tape for tape in all_groups if not tape.control and tape not in targets
                )
                if not targets:
                    control = GroupTape(
                        f"commit_{pc}", -1, 0, start_after_commit=last_commit, control=True
                    )
                    all_groups.append(control)
                    targets = (control,)
                last_commit = id(ins)
                next_step += 1
            else:
                targets = tuple(active[name] for name in sync_targets(op, args, active))
                if any(name not in seen_events for name in args["events"]):
                    raise ValueError("Wait for unknown event")
                waited_events.update(args["events"])
            if not targets:
                raise ValueError("Synchronization without active workgroup")
            for tape in targets:
                tape.commands.append((pc, ins))
            sync_participants[id(ins)] = targets
        else:
            event = args["event"]
            if event in seen_events:
                raise ValueError("Duplicate pipeline event")
            seen_events.add(event)
            if op in ("LD", "ST"):
                src, dst = args["src"], args["dst"]
                name = src["wg"] if src["space"] != "HBM" else dst["wg"]
            elif op == "MMA.ACC":
                name = args["acc"]["wg"]
            else:
                name = args["dst"]["wg"]
            tape = active.get(name)
            if tape is None:
                raise ValueError("Instruction for inactive workgroup")
            tape.commands.append((pc, ins))
    if active:
        raise ValueError("Unclosed pipeline workgroup")
    return ProgramQueues(
        tuple(all_groups), sync_participants, commit_closed, count, frozenset(waited_events)
    )


class TimeQueue:
    """Stable event queue ordered by cycle, kind, issue sequence, SM and PC."""

    def __init__(self):
        self._heap: list[tuple[int, int, int, int, int, int, Callable[[], None]]] = []
        self._serial = 0

    def push(
        self,
        cycle: int,
        kind: int,
        issue_sequence: int,
        sm: int,
        pc: int,
        action: Callable[[], None],
    ) -> None:
        if type(cycle) is not int or cycle < 0:
            raise ValueError("Invalid event cycle")
        heappush(self._heap, (cycle, kind, issue_sequence, sm, pc, self._serial, action))
        self._serial += 1

    def next_cycle(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def pop(self) -> int:
        cycle, _, _, _, _, _, action = heappop(self._heap)
        action()
        return cycle

    def __bool__(self) -> bool:
        return bool(self._heap)


def _overlap(left: dict, right: dict) -> bool:
    if "imm" in left or "imm" in right or left["space"] != right["space"]:
        return False
    if left["space"] != "HBM" and left["wg"] != right["wg"]:
        return False
    if left["space"] == "RF" and left["lane"] != right["lane"]:
        return False
    return _address_overlap(left, right)


@dataclass(eq=False)
class PendingTransfer:
    tape: GroupTape
    pc: int
    instruction: Instruction
    issue: int
    engine: int
    slot: int
    src: dict
    dst: dict
    payload: int
    source_finish: int
    line_count: int
    energy_runs: object
    energy: float = 0.0
    lines_done: int = 0
    max_line_finish: int = 0
    max_hbm_read_finish: int = 0
    finish: int | None = None


def _storage_key(operand: dict) -> tuple:
    space = operand["space"]
    if space == "HBM":
        return (space,)
    if space == "RF":
        return space, operand["wg"], operand["lane"]
    return space, operand["wg"]


class _PendingTransfers:
    """Pending DMA transfers indexed by the namespaces hazards can overlap."""

    def __init__(self):
        self._items: set[PendingTransfer] = set()
        self._destinations: dict[tuple, set[PendingTransfer]] = {}
        self._hbm_sources: set[PendingTransfer] = set()

    def __iter__(self):
        return iter(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def add(self, transfer: PendingTransfer) -> None:
        if transfer in self._items:
            return
        self._items.add(transfer)
        self._destinations.setdefault(_storage_key(transfer.dst), set()).add(transfer)
        if transfer.src["space"] == "HBM":
            self._hbm_sources.add(transfer)

    def remove(self, transfer: PendingTransfer) -> None:
        self._items.remove(transfer)
        key = _storage_key(transfer.dst)
        destinations = self._destinations[key]
        destinations.remove(transfer)
        if not destinations:
            del self._destinations[key]
        self._hbm_sources.discard(transfer)

    def has_hazard(self, reads, writes) -> bool:
        for operand in (*reads, *writes):
            if "imm" in operand:
                continue
            for transfer in self._destinations.get(_storage_key(operand), ()):
                if _overlap(operand, transfer.dst):
                    return True
        for operand in writes:
            if operand.get("space") != "HBM":
                continue
            for transfer in self._hbm_sources:
                if _overlap(operand, transfer.src):
                    return True
        return False


@dataclass(eq=False)
class _MulticastSource:
    owner: PendingTransfer
    line: int
    ready: int | None = None
    waiters: list[tuple[PendingTransfer, int, float]] = field(default_factory=list)
    destinations: set[tuple[int, str]] = field(default_factory=set)


def _resource_stats() -> dict:
    return {
        "sms": {},
        "shared": {
            "hbm_read_bytes": 0,
            "hbm_write_bytes": 0,
            "cache_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_bypass_reads": 0,
            "noc_bytes": 0,
            "busy_cycles": {},
        },
    }


class GlobalEventPipeline:
    """Issue independent workgroups and resolve memory at event arrival time."""

    def __init__(
        self,
        hardware: Hardware,
        program,
        hbm_words: int,
        step_requirements: dict[int, list[tuple[int, int]]] | None = None,
        trace: dict[str, dict[str, int]] | None = None,
        energy_trace: list[EnergyEvent] | None = None,
        *,
        hbm_races_validated: bool = False,
    ):
        from .pipeline import _MemoryTimes

        hardware.validate()
        if type(hbm_words) is not int or not 0 < hbm_words <= 2**29:
            raise ValueError("Invalid timed HBM size")
        self.hw = hardware
        self.queues = group_program(program)
        if any(
            not tape.control
            and (
                type(tape.sm) is not int
                or type(tape.shared_bytes) is not int
                or not 0 <= tape.sm < hardware.sm_count
                or tape.shared_bytes < 0
                or tape.shared_bytes % 4
                or tape.shared_bytes > hardware.shared_kib * 1024
            )
            for tape in self.queues.groups
        ):
            raise ValueError("Workgroup SM or shared quota exceeds hardware capacity")
        if not hbm_races_validated:
            from .hbm_race import validate_hbm_races

            validate_hbm_races(None, queues=self.queues)
        self.cost = cost_model()
        self.prices = self.cost["energy_pj"]
        self.hbm = _MemoryTimes.sized(hbm_words, sparse=True)
        self.step_requirements = step_requirements
        self.trace = trace
        self.energy_trace = energy_trace
        self.stats = _resource_stats()
        self.cache = TimedGlobalCache(hardware.cache_mib)
        self.group_states = {}
        self.resident_by_sm = [set() for _ in range(hardware.sm_count)]
        self.issue_free = [IntervalCalendar() for _ in range(hardware.sm_count)]
        self.rf_read = [[IntervalCalendar()] for _ in range(hardware.sm_count)]
        self.rf_write = [[IntervalCalendar()] for _ in range(hardware.sm_count)]
        self.tc = [[IntervalCalendar()] for _ in range(hardware.sm_count)]
        self.vec = [[IntervalCalendar()] for _ in range(hardware.sm_count)]
        self.sfu = [[IntervalCalendar()] for _ in range(hardware.sm_count)]
        self.reduce = [
            [IntervalCalendar()] if hardware.reduction_units else self.vec[sm]
            for sm in range(hardware.sm_count)
        ]
        slots = max(1, hardware.dma_depth)
        self.dma_slots: list[list[list[int | PendingTransfer]]] = [
            [[0] * slots for _ in range(hardware.dma_engines)] for _ in range(hardware.sm_count)
        ]
        self.dma_slice = [
            [IntervalCalendar() for _ in range(hardware.dma_engines)]
            for _ in range(hardware.sm_count)
        ]
        self.sm_noc_in = [
            ByteCalendar(hardware.sm_noc_bytes_per_cycle) for _ in range(hardware.sm_count)
        ]
        self.sm_noc_out = [
            ByteCalendar(hardware.sm_noc_bytes_per_cycle) for _ in range(hardware.sm_count)
        ]
        self.noc = ByteCalendar(hardware.noc_bytes_per_cycle)
        self.hbm_channels = [[IntervalCalendar()] for _ in range(hardware.hbm_channels)]
        self.shared_read = [
            [IntervalCalendar() for _ in range(hardware.shared_banks or 1)]
            for _ in range(hardware.sm_count)
        ]
        self.shared_write = [
            [IntervalCalendar() for _ in range(hardware.shared_banks or 1)]
            for _ in range(hardware.sm_count)
        ]
        self.queue = TimeQueue()
        self.pending = _PendingTransfers()
        self.commit_finish: dict[int, int] = {}
        self.events: dict[str, int] = {}
        self.energy_events: list[EnergyEvent] = []
        self.dynamic_energy = 0.0
        self.last_finish = 0
        self.hbm_read_bytes = 0
        self.hbm_write_bytes = 0
        self.issue_sequence = 0
        self.current_cycle = 0
        self.multicast_sources: dict[tuple[int, int], _MulticastSource] = {}
        self.multicast_arrivals: dict[tuple[int, int], int] = {}
        self.read_port, self.write_port = {
            "2R1W": (32, 16),
            "4R2W": (64, 32),
            "8R4W": (128, 64),
        }[hardware.rf_ports]
        self.rf_price = self.prices["rf_byte"][hardware.rf_ports]
        self.noc_price = (
            self.prices["noc_byte_fixed"]
            + self.prices["noc_byte_per_width"] * hardware.noc_bytes_per_cycle
        )

    def _stat_sm(self, sm: int) -> dict:
        return self.stats["sms"].setdefault(
            str(sm),
            {
                "instructions": 0,
                "dma_bytes": 0,
                "rf_read_bytes": 0,
                "rf_write_bytes": 0,
                "issue_wait_cycles": 0,
                "busy_cycles": {},
            },
        )

    def _busy(self, sm: int | None, resource: str, duration: int):
        target = self.stats["shared"] if sm is None else self._stat_sm(sm)
        busy = target["busy_cycles"]
        busy[resource] = busy.get(resource, 0) + duration

    def _record(self, name: str, issue: int, finish: int):
        if name in self.queues.waited_events:
            self.events[name] = finish
        if self.trace is not None:
            self.trace[name] = {"issue": issue, "finish": finish}
        self.last_finish = max(self.last_finish, finish)

    def _finish_energy(self, runs, energy: float):
        runs.flush()
        if abs(runs.total - energy) > max(1e-6, energy * 1e-9):
            raise AssertionError("Pipeline service energy does not match cost accounting")
        self.dynamic_energy += energy

    def _available_slot(self, sm: int, earliest: int):
        choices = []
        for engine in range(self.hw.dma_engines):
            for slot, held in enumerate(self.dma_slots[sm][engine]):
                if isinstance(held, PendingTransfer):
                    continue
                slice_ready = self.dma_slice[sm][engine].first_free(earliest + 4, 1) - 4
                choices.append((max(held, slice_ready), engine, slot))
        return min(choices) if choices else None

    def _pending_hazard(self, reads, writes) -> bool:
        return self.pending.has_hazard(reads, writes)

    def _operands(self, ins: Instruction):
        op, args = ins.op, ins.args
        if op in ("LD", "ST"):
            return [args["src"]], [args["dst"]]
        if op == "MMA.ACC":
            return [args["a"], args["b"], args["acc"]], [args["acc"]]
        if op == "VEC":
            return args["src"], [args["dst"]]
        return [args["src"]], [args["dst"]]

    def _preview(self, tape: GroupTape):
        from .pipeline import _max_destination, _max_write

        command = tape.next_command()
        if command is None:
            return None
        pc, ins = command
        op, args = ins.op, ins.args
        if op == "WG.BEGIN":
            if tape.predecessor is not None and tape.predecessor.end_cycle is None:
                return None
            if (
                tape.start_after_commit is not None
                and tape.start_after_commit not in self.commit_finish
            ):
                return None
            residents = self.resident_by_sm[tape.sm]
            if (
                len(residents) >= 4
                or args["shared_bytes"] + sum(peer.shared_bytes for peer in residents)
                > self.hw.shared_kib * 1024
            ):
                return None
            ready = max(
                tape.predecessor.end_cycle if tape.predecessor is not None else 0,
                self.commit_finish.get(tape.start_after_commit, 0),
            )
            selected = (tape,)
        elif op in ("WAIT", "BARRIER", "STEP.COMMIT"):
            selected = self.queues.sync_participants[id(ins)]
            if self.hw.dma_depth == 0 and any(
                transfer.tape in selected for transfer in self.pending
            ):
                return None
            if any(
                peer.next_command() is None or peer.next_command()[1] is not ins
                for peer in selected
            ):
                return None
            if tape is not selected[0]:
                return None
            if op == "STEP.COMMIT":
                if self.pending:
                    return None
                if (
                    self.step_requirements is not None
                    and args["step"] not in self.step_requirements
                ):
                    raise ValueError("Invalid step commitment")
                closed = self.queues.commit_closed[id(ins)]
                if any(peer.end_cycle is None for peer in closed):
                    return None
                if tape.control and (
                    tape.start_after_commit is not None
                    and tape.start_after_commit not in self.commit_finish
                ):
                    return None
                ranges = (
                    self.step_requirements.get(args["step"], [])
                    if self.step_requirements is not None
                    else []
                )
                if any(not self.hbm.written[lo:hi].all() for lo, hi in ranges):
                    raise ValueError("Step outputs incomplete")
                target = max((int(self.hbm.written[lo:hi].max()) for lo, hi in ranges), default=0)
                target = max(
                    target,
                    *(self.group_states[peer].last_finish for peer in selected if not peer.control),
                    *(peer.end_cycle for peer in closed),
                    self.commit_finish.get(tape.start_after_commit, 0),
                )
            else:
                names = args["events"]
                if any(name not in self.events for name in names):
                    return None
                target = max((self.events[name] for name in names), default=0)
                if op == "BARRIER":
                    if any(transfer.tape in selected for transfer in self.pending):
                        return None
                    target = max(
                        target, *(self.group_states[peer].last_finish for peer in selected)
                    )
            ready = max(
                [target]
                + [self.group_states[peer].next_issue for peer in selected if not peer.control]
            )
        else:
            group = self.group_states[tape]
            selected = (tape,)
            if self.hw.dma_depth == 0 and any(transfer.tape is tape for transfer in self.pending):
                return None
            if op in ("LD", "ST") and all(
                isinstance(held, PendingTransfer)
                for engine in self.dma_slots[tape.sm]
                for held in engine
            ):
                return None
            if op == "WG.END":
                if any(transfer.tape is tape for transfer in self.pending):
                    return None
                ready = max(group.next_issue, group.last_finish)
            else:
                reads, writes = self._operands(ins)
                if self._pending_hazard(reads, writes):
                    return None
                ready = max(
                    group.next_issue,
                    *(_max_write(group, self.hbm, operand) for operand in reads),
                    *(_max_destination(group, self.hbm, operand) for operand in writes),
                )
                if op in ("LD", "ST"):
                    slot = self._available_slot(tape.sm, ready)
                    if slot is None:
                        return None
                    ready = max(ready, slot[0])
        sms = {peer.sm for peer in selected if not peer.control}
        begin = max(ready, self.current_cycle)
        while sms:
            candidate = max(self.issue_free[sm].first_free(begin, 1) for sm in sms)
            if candidate == begin:
                break
            begin = candidate
        if op in ("LD", "ST"):
            # A later issue cycle can overlap a previously reserved DMA slice.
            # Recheck both calendars until neither moves the transfer.
            while True:
                slot = self._available_slot(tape.sm, begin)
                if slot is None:
                    return None
                candidate = max(begin, slot[0])
                candidate = max(
                    candidate,
                    *(self.issue_free[sm].first_free(candidate, 1) for sm in sms),
                )
                if candidate == begin:
                    break
                begin = candidate
        return begin, pc, tape, selected

    def _issue(self, preview):
        from .pipeline import _GroupTimes, _MemoryTimes

        begin, pc, tape, selected = preview
        if begin < self.current_cycle:
            raise AssertionError("Instruction issued before global event time")
        self.current_cycle = begin
        ins = tape.next_command()[1]
        op = ins.op
        for sm in {peer.sm for peer in selected if not peer.control}:
            actual, _ = self.issue_free[sm].reserve(begin, 1)
            if actual != begin:
                raise AssertionError("Issue calendar changed after preview")
            self._busy(sm, "issue", 1)
        self.issue_sequence += 1
        sequence = self.issue_sequence
        if op == "WG.BEGIN":
            group = _GroupTimes(
                tape.name,
                tape.sm,
                16384 // self.hw.vector_lanes,
                begin + 1,
                begin + 1,
                _MemoryTimes.sized(65536 // 4),
                _MemoryTimes.sized(tape.shared_bytes // 4),
            )
            self.group_states[tape] = group
            self.resident_by_sm[tape.sm].add(tape)
            self.last_finish = max(self.last_finish, begin + 1)
        elif op == "WG.END":
            self.resident_by_sm[tape.sm].remove(tape)
            del self.group_states[tape]
            tape.end_cycle = begin + 1
            self.last_finish = max(self.last_finish, begin + 1)
        elif op in ("WAIT", "BARRIER", "STEP.COMMIT"):
            for peer in selected:
                if not peer.control:
                    self.group_states[peer].next_issue = begin + 1
            if op == "STEP.COMMIT":
                self.commit_finish[id(ins)] = begin + 1
            self.last_finish = max(self.last_finish, begin + 1)
        else:
            group = self.group_states[tape]
            self._stat_sm(tape.sm)["issue_wait_cycles"] += max(0, begin - group.next_issue)
            group.next_issue = begin + 1
            self._stat_sm(tape.sm)["instructions"] += 1
            if op in ("LD", "ST"):
                self._issue_transfer(tape, pc, ins, begin, sequence)
            else:
                self._issue_compute(tape, pc, ins, begin)
        for peer in selected:
            peer.commands[peer.cursor] = None
            peer.cursor += 1

    def _issue_compute(self, tape: GroupTape, pc: int, ins: Instruction, begin: int):
        from .pipeline import _EnergyRuns, _mark, _reserve

        group = self.group_states[tape]
        sm = tape.sm
        op, args = ins.op, ins.args
        runs = _EnergyRuns(self.energy_events)
        if op == "MMA.ACC":
            reads, writes = [args["a"], args["b"], args["acc"]], [args["acc"]]
            demand = mma_service(self.hw, args["m"], args["n"], args["k"])
            read_bytes, write_bytes = demand.rf_read_bytes, demand.rf_write_bytes
            cycles, unit, resource = demand.compute_cycles, self.tc[sm], "tc"
            arithmetic = demand.arithmetic_pj
        elif op == "VEC":
            reads, writes = args["src"], [args["dst"]]
            count = args["dst"]["count"]
            read_bytes = 4 * sum(item["count"] for item in reads if "space" in item)
            write_bytes = 4 * count
            cycles, unit, resource = vector_cycles(count, self.hw.vector_lanes), self.vec[sm], "vec"
            arithmetic = (
                count * self.prices["vector_fma" if args["kind"] == "fma" else "vector_other"]
            )
        elif op == "REDUCE":
            reads, writes = [args["src"]], [args["dst"]]
            count = args["src"]["count"]
            read_bytes, write_bytes = 4 * count, 4
            cycles = reduction_cycles(count, self.hw.vector_lanes, self.hw.reduction_units)
            unit = self.reduce[sm]
            resource = "reduce" if self.hw.reduction_units else "vec"
            arithmetic = (
                max(0, count - 1)
                * self.prices[
                    "reduction_dedicated_merge"
                    if self.hw.reduction_units
                    else "reduction_vector_merge"
                ]
            )
        elif op == "SFU":
            reads, writes = [args["src"]], [args["dst"]]
            count = args["src"]["count"]
            read_bytes = write_bytes = 4 * count
            cycles, unit, resource = sfu_cycles(count, self.hw.sfu_lanes), self.sfu[sm], "sfu"
            arithmetic = count * self.prices["sfu"]
        else:
            raise ValueError(f"Unknown pipeline opcode: {op}")
        read_duration = ceil(read_bytes / self.read_port)
        read_end = _reserve(self.rf_read[sm], begin + 1, read_duration)
        compute_end = _reserve(unit, read_end, cycles)
        write_duration = ceil(write_bytes / self.write_port)
        finish = _reserve(self.rf_write[sm], compute_end, write_duration)
        rf_energy = (read_bytes + write_bytes) * self.rf_price
        self._stat_sm(sm)["rf_read_bytes"] += read_bytes
        self._stat_sm(sm)["rf_write_bytes"] += write_bytes
        self._busy(sm, "rf_read", read_duration)
        self._busy(sm, resource, cycles)
        self._busy(sm, "rf_write", write_duration)
        if read_bytes:
            runs.add(
                ("rf_read", sm), read_end - read_duration, read_end, read_bytes * self.rf_price
            )
        runs.add((resource, sm), compute_end - cycles, compute_end, arithmetic)
        runs.add(("rf_write", sm), finish - write_duration, finish, write_bytes * self.rf_price)
        _mark(group, self.hbm, reads, writes, read_end, finish)
        group.last_finish = max(group.last_finish, finish)
        self._record(args["event"], begin, finish)
        self._finish_energy(runs, arithmetic + rf_energy)

    def _shared_service(self, tape: GroupTape, operand: dict, earliest: int, write: bool, runs):
        if not self.hw.shared_banks:
            raise ValueError("Shared access on hardware without shared memory")
        banks = [0] * self.hw.shared_banks
        for word in _positions(operand):
            banks[(word // 16) % self.hw.shared_banks] += 4
        rate = 16 if write or self.hw.shared_ports == "1R1W" else 32
        calendars = self.shared_write if write else self.shared_read
        finish = earliest
        for bank, nbytes in enumerate(banks):
            if not nbytes:
                continue
            start, end = calendars[tape.sm][bank].reserve(earliest, ceil(nbytes / rate))
            runs.add(
                ("shared_write" if write else "shared_read", tape.sm, bank),
                start,
                end,
                nbytes * self.prices["shared_byte"][self.hw.shared_ports],
            )
            self._busy(tape.sm, "shared_write" if write else "shared_read", end - start)
            finish = max(finish, end)
        return finish

    def _issue_transfer(
        self, tape: GroupTape, pc: int, ins: Instruction, begin: int, sequence: int
    ):
        from .pipeline import _EnergyRuns, _mark, _reserve

        group = self.group_states[tape]
        sm = tape.sm
        src, dst = ins.args["src"], ins.args["dst"]
        slot_choice = self._available_slot(sm, begin)
        if slot_choice is None or slot_choice[0] > begin:
            raise AssertionError("DMA slot changed after issue preview")
        _, engine, slot = slot_choice
        payload = src["count"] * 4
        runs = _EnergyRuns(self.energy_events)
        energy = 0.0
        time = begin + 4
        if src["space"] == "RF":
            duration = ceil(payload / self.read_port)
            time = _reserve(self.rf_read[sm], time, duration)
            energy += payload * self.rf_price
            runs.add(("rf_read", sm), time - duration, time, payload * self.rf_price)
            self._stat_sm(sm)["rf_read_bytes"] += payload
            self._busy(sm, "rf_read", duration)
        elif src["space"] == "SH":
            time = self._shared_service(tape, src, time, False, runs)
            energy += payload * self.prices["shared_byte"][self.hw.shared_ports]
        source_finish = time
        if dst["space"] == "HBM":
            # HBM writes remain pending after the local RF/SH source read.
            # Publish its read hazard now so a later local write cannot
            # overwrite bytes while this asynchronous read is in service.
            _mark(group, self.hbm, [src], [], source_finish, source_finish)
        if "HBM" not in (src["space"], dst["space"]):
            if dst["space"] == "RF":
                duration = ceil(payload / self.write_port)
                time = _reserve(self.rf_write[sm], time, duration)
                energy += payload * self.rf_price
                runs.add(("rf_write", sm), time - duration, time, payload * self.rf_price)
                self._stat_sm(sm)["rf_write_bytes"] += payload
                self._busy(sm, "rf_write", duration)
            elif dst["space"] == "SH":
                time = self._shared_service(tape, dst, time, True, runs)
                energy += payload * self.prices["shared_byte"][self.hw.shared_ports]
            finish = time
            self.dma_slots[sm][engine][slot] = finish
            if self.hw.dma_depth == 0:
                group.next_issue = finish
            _mark(group, self.hbm, [src], [dst], source_finish, finish)
            group.last_finish = max(group.last_finish, finish)
            self._record(ins.args["event"], begin, finish)
            self._finish_energy(runs, energy)
            return

        address = src if src["space"] == "HBM" else dst
        lines = _lines(address)
        transfer = PendingTransfer(
            tape, pc, ins, begin, engine, slot, src, dst, payload, source_finish, len(lines), runs
        )
        transfer.energy = (
            energy + len(lines) * 64 * self.noc_price + len(lines) * self.prices["dma_slice"]
        )
        self._stat_sm(sm)["dma_bytes"] += payload
        self.pending.add(transfer)
        self.dma_slots[sm][engine][slot] = transfer
        network_energy_per_byte = self.noc_price
        for line in lines:
            if self.hw.multicast and src["space"] == "HBM":
                key = (begin, line)
                self.multicast_arrivals[key] = self.multicast_arrivals.get(key, 0) + 1
            dma_start, dma_end = self.dma_slice[sm][engine].reserve(time, 1)
            runs.add(("dma", sm, engine), dma_start, dma_end, self.prices["dma_slice"])
            self._busy(sm, "dma", 1)
            self.queue.push(
                dma_start,
                0,
                sequence,
                sm,
                pc,
                lambda line=line, start=dma_start, transfer=transfer, price=network_energy_per_byte: (
                    self._line_arrival(transfer, line, start, price)
                ),
            )

    def _hbm_read_line(self, owner: PendingTransfer, line: int, earliest: int) -> int:
        from .pipeline import _reserve

        channel = (line * 64 // 256) % self.hw.hbm_channels
        finish = _reserve(self.hbm_channels[channel], earliest, 2)
        energy = 64 * self.prices["hbm_byte"]
        owner.energy += energy
        owner.energy_runs.add(("hbm_read", channel), finish - 2, finish, energy)
        self.hbm_read_bytes += 64
        self.stats["shared"]["hbm_read_bytes"] += 64
        self._busy(None, f"hbm_channel_{channel}", 2)
        return finish

    def _hbm_read(self, source: _MulticastSource, earliest: int) -> int:
        return self._hbm_read_line(source.owner, source.line, earliest)

    def _cache_source(self, source: _MulticastSource, pending_lookup):
        line = source.line
        ready, hit, _ = self.cache.resolve_lookup(
            pending_lookup, lambda _address, earliest: self._hbm_read(source, earliest)
        )
        access = self.cache.last_access
        assert access is not None and access.lookup_cycle is not None
        bank = self.cache.locate(line * 64)[1]
        query_energy = self.prices["cache_query_or_invalidate"]
        source.owner.energy += query_energy
        source.owner.energy_runs.add(
            ("cache_query", bank), access.lookup_cycle, access.lookup_cycle + 1, query_energy
        )
        self.stats["shared"]["cache_queries"] += 1
        self._busy(None, f"cache_bank_{bank}", 1)
        if hit:
            read_energy = 64 * self.prices["cache_read_byte"]
            source.owner.energy += read_energy
            source.owner.energy_runs.add(
                ("cache_read", bank), access.lookup_cycle, ready, read_energy
            )
            self.stats["shared"]["cache_hits"] += 1
        else:
            assert access.fill_cycle is not None
            fill_energy = 64 * self.prices["cache_fill_byte"]
            source.owner.energy += fill_energy
            source.owner.energy_runs.add(
                ("cache_fill", bank), access.fill_cycle, access.fill_cycle + 1, fill_energy
            )
            self.stats["shared"]["cache_misses"] += 1
            self._busy(None, f"cache_bank_{bank}", 1)
        self._source_ready(source, ready)

    def _source_ready(self, source: _MulticastSource, ready: int):
        source.ready = ready
        for transfer, arrival, price in source.waiters:
            self._network_read(transfer, ready, arrival, price)
        source.waiters.clear()

    def _network_read(
        self, transfer: PendingTransfer, source_ready: int, arrival: int, energy_per_byte: float
    ):
        sm = transfer.tape.sm
        finish = reserve_linked_bytes(
            (self.noc, self.sm_noc_in[sm]),
            max(source_ready, arrival),
            64,
            on_service=lambda cycle, take: transfer.energy_runs.add(
                ("noc_in", sm), cycle, cycle + 1, take * energy_per_byte
            ),
        )[1]
        self.stats["shared"]["noc_bytes"] += 64
        transfer.max_hbm_read_finish = max(transfer.max_hbm_read_finish, source_ready)
        self._complete_line(transfer, finish)

    def _line_arrival(
        self, transfer: PendingTransfer, line: int, arrival: int, network_energy_per_byte: float
    ):
        from .pipeline import _reserve

        sm = transfer.tape.sm
        if transfer.src["space"] == "HBM":
            if not self.hw.cache_mib and not self.hw.multicast:
                ready = self._hbm_read_line(transfer, line, arrival)
                traffic = self.cache.traffic
                traffic.bypass_reads += 1
                traffic.hbm_read_bytes += 64
                self.cache.last_access = CacheAccess(None, None, ready, False, True)
                self.stats["shared"]["cache_bypass_reads"] += 1
                self._network_read(transfer, ready, arrival, network_energy_per_byte)
                return
            source_key = (transfer.issue, line)
            destination = (sm, transfer.tape.name)
            source = self.multicast_sources.get(source_key) if self.hw.multicast else None
            if source is not None and destination not in source.destinations:
                source.destinations.add(destination)
                if source.ready is None:
                    source.waiters.append((transfer, arrival, network_energy_per_byte))
                else:
                    self._network_read(transfer, source.ready, arrival, network_energy_per_byte)
                self._multicast_arrived(source_key)
                return
            source = _MulticastSource(transfer, line)
            source.destinations.add(destination)
            source.waiters.append((transfer, arrival, network_energy_per_byte))
            if self.hw.multicast:
                self.multicast_sources[source_key] = source
            if self.hw.cache_mib:
                pending_lookup = self.cache.reserve_lookup(line * 64, arrival)
                self.queue.push(
                    pending_lookup.cycle,
                    1,
                    transfer.issue,
                    sm,
                    transfer.pc,
                    lambda source=source, pending=pending_lookup: self._cache_source(
                        source, pending
                    ),
                )
            else:
                ready, _, _ = self.cache.request_read(
                    line * 64, arrival, lambda _address, earliest: self._hbm_read(source, earliest)
                )
                self.stats["shared"]["cache_bypass_reads"] += 1
                self._source_ready(source, ready)
            self._multicast_arrived(source_key)
            return

        finish = reserve_linked_bytes(
            (self.noc, self.sm_noc_out[sm]),
            arrival,
            64,
            on_service=lambda cycle, take: transfer.energy_runs.add(
                ("noc_out", sm), cycle, cycle + 1, take * network_energy_per_byte
            ),
        )[1]
        self.stats["shared"]["noc_bytes"] += 64
        channel = (line * 64 // 256) % self.hw.hbm_channels
        hbm_done = _reserve(self.hbm_channels[channel], finish, 2)
        hbm_energy = transfer.payload / transfer.line_count * self.prices["hbm_byte"]
        transfer.energy += hbm_energy
        transfer.energy_runs.add(("hbm_write", channel), hbm_done - 2, hbm_done, hbm_energy)
        self._busy(None, f"hbm_channel_{channel}", 2)
        if self.hw.cache_mib:
            # The completed HBM write immediately invalidates its line. This
            # consumes tag energy in the final write cycle, but no read/fill
            # cache bank slot.
            self.cache.schedule_write_invalidate(line * 64, hbm_done)
            invalidate_energy = self.prices["cache_query_or_invalidate"]
            transfer.energy += invalidate_energy
            transfer.energy_runs.add(
                ("cache_invalidate", self.cache.locate(line * 64)[1]),
                hbm_done - 1,
                hbm_done,
                invalidate_energy,
            )
        self._complete_line(transfer, hbm_done)

    def _multicast_arrived(self, key: tuple[int, int]) -> None:
        if not self.hw.multicast:
            return
        remaining = self.multicast_arrivals[key] - 1
        if remaining:
            self.multicast_arrivals[key] = remaining
        else:
            del self.multicast_arrivals[key]
            self.multicast_sources.pop(key, None)

    def _complete_line(self, transfer: PendingTransfer, finish: int):
        transfer.lines_done += 1
        transfer.max_line_finish = max(transfer.max_line_finish, finish)
        if transfer.lines_done == transfer.line_count:
            self._finish_transfer(transfer)

    def _finish_transfer(self, transfer: PendingTransfer):
        from .pipeline import _mark, _reserve

        tape = transfer.tape
        group = self.group_states[tape]
        sm = tape.sm
        time = transfer.max_line_finish
        dst = transfer.dst
        if dst["space"] == "RF":
            duration = ceil(transfer.payload / self.write_port)
            time = _reserve(self.rf_write[sm], time, duration)
            transfer.energy += transfer.payload * self.rf_price
            transfer.energy_runs.add(
                ("rf_write", sm),
                time - duration,
                time,
                transfer.payload * self.rf_price,
            )
            self._stat_sm(sm)["rf_write_bytes"] += transfer.payload
            self._busy(sm, "rf_write", duration)
        elif dst["space"] == "SH":
            time = self._shared_service(tape, dst, time, True, transfer.energy_runs)
            transfer.energy += transfer.payload * self.prices["shared_byte"][self.hw.shared_ports]
        finish = time
        transfer.finish = finish
        self.dma_slots[sm][transfer.engine][transfer.slot] = finish
        if self.hw.dma_depth == 0:
            group.next_issue = max(group.next_issue, finish)
        read_finish = (
            transfer.max_hbm_read_finish
            if transfer.src["space"] == "HBM"
            else transfer.source_finish
        )
        _mark(group, self.hbm, [transfer.src], [dst], read_finish, finish)
        group.last_finish = max(group.last_finish, finish)
        if dst["space"] == "HBM":
            self.hbm_write_bytes += transfer.payload
            self.stats["shared"]["hbm_write_bytes"] += transfer.payload
        self._record(transfer.instruction.args["event"], transfer.issue, finish)
        self._finish_energy(transfer.energy_runs, transfer.energy)
        self.pending.remove(transfer)

    def _make_timeline(self, static_power: float, peak_window_w: float) -> dict:
        from .resource_timeline import ResourceTimeline

        timeline = ResourceTimeline(self.last_finish)
        for sm in range(self.hw.sm_count):
            if str(sm) not in self.stats["sms"]:
                continue
            exclusive = {
                "issue": [self.issue_free[sm]],
                "rf_read": self.rf_read[sm],
                "rf_write": self.rf_write[sm],
                "tc": self.tc[sm],
                "vec": self.vec[sm],
                "sfu": self.sfu[sm],
                "dma": self.dma_slice[sm],
            }
            if self.hw.reduction_units:
                exclusive["reduce"] = self.reduce[sm]
            if self.hw.shared_banks:
                exclusive["shared_read"] = self.shared_read[sm]
                exclusive["shared_write"] = self.shared_write[sm]
            for resource, calendars in exclusive.items():
                timeline.add_exclusive(sm, resource, calendars)
            timeline.add_bandwidth(sm, "noc_in", self.sm_noc_in[sm])
            timeline.add_bandwidth(sm, "noc_out", self.sm_noc_out[sm])
        timeline.add_bandwidth(None, "noc", self.noc)
        for channel, calendars in enumerate(self.hbm_channels):
            timeline.add_exclusive(None, f"hbm_channel_{channel}", calendars)
        if self.hw.cache_mib:
            for bank, calendar in enumerate(self.cache._bank_busy):
                timeline.add_exclusive(None, f"cache_bank_{bank}", [calendar])
        result = timeline.finish(
            self.energy_events,
            self.dynamic_energy,
            static_power,
            self.cost["clock_hz"],
            self.cost["power_window_cycles"],
            peak_window_w,
        )
        for scope, totals in self.stats["sms"].items():
            for resource, total in totals["busy_cycles"].items():
                if sum(result["sms"][scope]["busy_cycles"][resource]) != total:
                    raise AssertionError("SM timeline busy cycles do not conserve reservations")
        for resource, total in self.stats["shared"]["busy_cycles"].items():
            if sum(result["shared"]["busy_cycles"][resource]) != total:
                raise AssertionError("Shared timeline busy cycles do not conserve reservations")
        if sum(result["shared"]["bytes"]["noc"]) != self.stats["shared"]["noc_bytes"]:
            raise AssertionError("NoC timeline bytes do not conserve transfers")
        return result

    def run(self) -> PerfResult:
        while True:
            candidates = [
                candidate
                for tape in self.queues.groups
                if (candidate := self._preview(tape)) is not None
            ]
            event_cycle = self.queue.next_cycle()
            if not candidates and event_cycle is None:
                if all(tape.next_command() is None for tape in self.queues.groups):
                    break
                blocked = [
                    (tape.name, tape.next_command()[1].op)
                    for tape in self.queues.groups
                    if tape.next_command() is not None
                ]
                raise ValueError(f"Pipeline deadlock: {blocked[:8]}")
            next_instruction = (
                min(candidates, key=lambda item: (item[0], item[1], item[2].sm))
                if candidates
                else None
            )
            if event_cycle is not None and (
                next_instruction is None or event_cycle <= next_instruction[0]
            ):
                self.current_cycle = max(self.current_cycle, self.queue.pop())
            else:
                self._issue(next_instruction)
        if self.pending:
            raise AssertionError("Pending DMA after all program commands")
        if self.step_requirements is not None and len(self.commit_finish) != len(
            self.step_requirements
        ):
            raise ValueError("Required step commitments incomplete")
        self.cache.drain(self.last_finish)
        if self.cache.traffic.hits + self.cache.traffic.misses != self.cache.traffic.queries:
            raise AssertionError("Cache query counters do not balance")
        if not self.hw.cache_mib and self.cache.traffic.bypass_reads * 64 != self.hbm_read_bytes:
            raise AssertionError("Bypassed read lines do not balance HBM traffic")
        for sm in range(self.hw.sm_count):
            stats = self.stats["sms"].get(str(sm))
            if stats is None:
                continue
            busy = stats["busy_cycles"]
            busy["issue"] = self.issue_free[sm].busy_cycles()
            busy["rf_read"] = self.rf_read[sm][0].busy_cycles()
            busy["rf_write"] = self.rf_write[sm][0].busy_cycles()
            busy["tc"] = self.tc[sm][0].busy_cycles()
            busy["vec"] = self.vec[sm][0].busy_cycles()
            busy["sfu"] = self.sfu[sm][0].busy_cycles()
            if self.hw.reduction_units:
                busy["reduce"] = self.reduce[sm][0].busy_cycles()
            busy["dma"] = sum(item.busy_cycles() for item in self.dma_slice[sm])
            busy["noc_in"] = self.sm_noc_in[sm].busy_cycles()
            busy["noc_out"] = self.sm_noc_out[sm].busy_cycles()
            if self.hw.shared_banks:
                busy["shared_read"] = sum(item.busy_cycles() for item in self.shared_read[sm])
                busy["shared_write"] = sum(item.busy_cycles() for item in self.shared_write[sm])
            stats["busy_cycles_kind"] = "sum_of_entity_cycles"
            capacities = {resource: 1 for resource in busy}
            capacities["dma"] = self.hw.dma_engines
            if self.hw.shared_banks:
                capacities["shared_read"] = self.hw.shared_banks
                capacities["shared_write"] = self.hw.shared_banks
            stats["entity_count"] = capacities
            stats["utilization"] = {
                resource: used / (self.last_finish * capacities[resource])
                for resource, used in busy.items()
            }
        shared = self.stats["shared"]
        shared["busy_cycles"]["noc"] = self.noc.busy_cycles()
        for channel, calendars in enumerate(self.hbm_channels):
            shared["busy_cycles"][f"hbm_channel_{channel}"] = calendars[0].busy_cycles()
        for bank, calendar in enumerate(self.cache._bank_busy):
            if self.hw.cache_mib:
                shared["busy_cycles"][f"cache_bank_{bank}"] = calendar.busy_cycles()
        shared["busy_cycles_kind"] = "sum_of_entity_cycles"
        shared["entity_count"] = {resource: 1 for resource in shared["busy_cycles"]}
        shared["utilization"] = {
            resource: used / self.last_finish for resource, used in shared["busy_cycles"].items()
        }
        shared["cache_queries"] = self.cache.traffic.queries
        shared["cache_hits"] = self.cache.traffic.hits
        shared["cache_misses"] = self.cache.traffic.misses
        shared["cache_bypass_reads"] = self.cache.traffic.bypass_reads
        shared["cache_fills"] = self.cache.traffic.fills
        shared["cache_invalidations"] = self.cache.traffic.invalidations
        shared["cache_read_bytes"] = self.cache.traffic.cache_read_bytes
        shared["cache_fill_bytes"] = self.cache.traffic.cache_fill_bytes
        shared["hbm_read_bytes"] = self.hbm_read_bytes
        shared["hbm_write_bytes"] = self.hbm_write_bytes
        shared["critical_completion"] = {"cycle": self.last_finish}
        shared["reservation_intervals"] = {
            "noc": self.noc.interval_count,
            "hbm_channels": [item[0].interval_count for item in self.hbm_channels],
            "sm_noc_in": [item.interval_count for item in self.sm_noc_in],
            "sm_noc_out": [item.interval_count for item in self.sm_noc_out],
            "energy_events": len(self.energy_events),
        }
        area = self.hw.area_mm2()
        seconds = self.last_finish / self.cost["clock_hz"]
        static_power = self.hw.static_power_w()
        peak_window_w, peak_instant_w = power_profile_w(area, self.energy_events)
        if self.energy_trace is not None:
            self.energy_trace.extend(self.energy_events)
        if abs(sum(event.dynamic_pj for event in self.energy_events) - self.dynamic_energy) > max(
            1e-6, self.dynamic_energy * 1e-9
        ):
            raise AssertionError("Dynamic energy events do not balance")
        self.stats["timeline"] = self._make_timeline(static_power, peak_window_w)
        return PerfResult(
            cycles=self.last_finish,
            dynamic_energy_pj=self.dynamic_energy,
            total_energy_pj=self.dynamic_energy + static_power * seconds * 1e12,
            average_power_w=self.dynamic_energy * 1e-12 / seconds + static_power,
            peak_window_power_w=peak_window_w,
            peak_power_upper_bound_w=peak_instant_w,
            area_mm2=area,
            instruction_count=self.queues.instruction_count,
            hbm_read_bytes=self.hbm_read_bytes,
            hbm_write_bytes=self.hbm_write_bytes,
            cache_hits=self.cache.traffic.hits,
            cache_misses=self.cache.traffic.misses,
            resource_stats=self.stats,
        )


def estimate_global_events(
    hardware: Hardware,
    program,
    hbm_words: int,
    step_requirements: dict[int, list[tuple[int, int]]] | None = None,
    trace: dict[str, dict[str, int]] | None = None,
    energy_trace: list[EnergyEvent] | None = None,
    *,
    hbm_races_validated: bool = False,
) -> PerfResult:
    return GlobalEventPipeline(
        hardware,
        program,
        hbm_words,
        step_requirements,
        trace,
        energy_trace,
        hbm_races_validated=hbm_races_validated,
    ).run()
