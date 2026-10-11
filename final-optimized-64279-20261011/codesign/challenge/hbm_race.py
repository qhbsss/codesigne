"""Reject unordered, conflicting HBM accesses in a parsed microprogram.

The source order interleaves independent workgroup tapes; it is not a memory
order between them.  An event wait orders the named operation's completion,
while a barrier or commit completes all earlier operations in its scope.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .address_views import bounds as _bounds
from .address_views import overlap as _overlap
from .isa import Instruction, iter_parse
from .pipeline_events import ProgramQueues, group_program


@dataclass(frozen=True)
class _Knowledge:
    prefixes: dict[int, int]
    # Per-tape bitmaps keep long unsynchronized programs from retaining a
    # Python object for every individually completed event in each snapshot.
    events: dict[int, int]

    def contains(self, identity: tuple[int, int]) -> bool:
        tape, number = identity
        return self.prefixes.get(tape, 0) >= number or bool(
            self.events.get(tape, 0) & (1 << number)
        )


_EMPTY = _Knowledge({}, {})


def _join(*items: _Knowledge) -> _Knowledge:
    prefixes: dict[int, int] = {}
    events: dict[int, int] = {}
    for item in items:
        for tape, number in item.prefixes.items():
            prefixes[tape] = max(prefixes.get(tape, 0), number)
        for tape, bits in item.events.items():
            events[tape] = events.get(tape, 0) | bits
    # A completed tape prefix already proves every earlier event. Also fold
    # any consecutive known events into that prefix: local write dependencies
    # often form a long chain without an explicit barrier or commit.
    remaining: dict[int, int] = {}
    for tape, bits in events.items():
        prefix = prefixes.get(tape, 0)
        pending = bits >> (prefix + 1)
        consecutive = (pending ^ (pending + 1)).bit_length() - 1
        if consecutive:
            prefix += consecutive
            prefixes[tape] = prefix
        if prefix:
            bits = (bits >> (prefix + 1)) << (prefix + 1)
        if bits:
            remaining[tape] = bits
    return _Knowledge(prefixes, remaining)


def _completed(knowledge: _Knowledge, tape: int, number: int) -> _Knowledge:
    return _join(knowledge, _Knowledge({tape: number}, {}))


def _event(knowledge: _Knowledge, identity: tuple[int, int]) -> _Knowledge:
    tape, number = identity
    return _join(knowledge, _Knowledge({}, {tape: 1 << number}))


@dataclass(eq=False)
class _Access:
    pc: int
    tape: int
    identity: tuple[int, int]
    descriptor: dict
    write: bool
    lo: int
    hi: int
    completion: _Knowledge | None = None


@dataclass(eq=False)
class _LocalWrite:
    pc: int
    descriptor: dict
    lo: int
    hi: int
    completion: _Knowledge


@dataclass
class _Node:
    access: _Access | _LocalWrite
    priority: int
    left: _Node | None = None
    right: _Node | None = None
    max_hi: int = 0

    def __post_init__(self):
        self.max_hi = self.access.hi


def _refresh(node: _Node) -> _Node:
    node.max_hi = max(
        node.access.hi,
        node.left.max_hi if node.left else 0,
        node.right.max_hi if node.right else 0,
    )
    return node


def _rotate_left(node: _Node) -> _Node:
    child = node.right
    assert child is not None
    node.right = child.left
    child.left = _refresh(node)
    return _refresh(child)


def _rotate_right(node: _Node) -> _Node:
    child = node.left
    assert child is not None
    node.left = child.right
    child.right = _refresh(node)
    return _refresh(child)


def _key(access: _Access | _LocalWrite) -> tuple[int, int]:
    return access.lo, access.pc


def _insert(root: _Node | None, node: _Node) -> _Node:
    if root is None:
        return node
    if _key(node.access) < _key(root.access):
        root.left = _insert(root.left, node)
        if root.left.priority < root.priority:
            root = _rotate_right(root)
    else:
        root.right = _insert(root.right, node)
        if root.right.priority < root.priority:
            root = _rotate_left(root)
    return _refresh(root)


def _erase(root: _Node | None, key: tuple[int, int]) -> _Node | None:
    if root is None:
        return None
    current = _key(root.access)
    if key < current:
        root.left = _erase(root.left, key)
    elif key > current:
        root.right = _erase(root.right, key)
    elif root.left is None:
        return root.right
    elif root.right is None:
        return root.left
    elif root.left.priority < root.right.priority:
        root = _rotate_right(root)
        root.right = _erase(root.right, key)
    else:
        root = _rotate_left(root)
        root.left = _erase(root.left, key)
    return _refresh(root)


def _candidates(root: _Node | None, lo: int, hi: int):
    if root is None or root.max_hi <= lo:
        return
    if root.left is not None:
        yield from _candidates(root.left, lo, hi)
    if root.access.lo < hi and root.access.hi > lo:
        yield root.access
    if root.access.lo < hi and root.right is not None:
        yield from _candidates(root.right, lo, hi)


def _priority(pc: int) -> int:
    # Reproducible mixing keeps monotone addresses from skewing the treap.
    value = pc + 0x9E3779B97F4A7C15
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _operands(ins: Instruction) -> tuple[list[dict], list[dict]]:
    op, args = ins.op, ins.args
    if op in {"LD", "ST"}:
        return [args["src"]], [args["dst"]]
    if op == "MMA.ACC":
        return [args["a"], args["b"], args["acc"]], [args["acc"]]
    if op == "VEC":
        return args["src"], [args["dst"]]
    return [args["src"]], [args["dst"]]


def _local_key(tape: int, operand: dict) -> tuple[int, str, int] | None:
    if not isinstance(operand, dict):
        raise ValueError("Invalid memory operand")
    if operand.get("space") not in {"RF", "SH"}:
        return None
    return tape, operand["space"], operand["lane"]


def _graph(program: tuple[Instruction, ...], queues: ProgramQueues | None):
    if queues is None:
        queues = group_program(program)
    tape_ids = {tape: index for index, tape in enumerate(queues.groups)}
    owners: list[tuple[int, ...]] = [()] * len(program)
    for tape, index in tape_ids.items():
        for command in tape.commands:
            assert command is not None
            pc, _ = command
            owners[pc] = (*owners[pc], index)
    edges: list[list[int]] = [[] for _ in program]
    indegree = [0] * len(program)

    def edge(before: int, after: int) -> None:
        edges[before].append(after)
        indegree[after] += 1

    for tape in queues.groups:
        commands = tape.commands
        for before, after in zip(commands, commands[1:]):
            assert before is not None and after is not None
            edge(before[0], after[0])
        if tape.predecessor is not None:
            predecessor_end = tape.predecessor.commands[-1]
            begin = commands[0]
            assert predecessor_end is not None and begin is not None
            edge(predecessor_end[0], begin[0])

    event_pc = {
        ins.args["event"]: pc
        for pc, ins in enumerate(program)
        if ins.op not in {"WG.BEGIN", "WG.END", "WAIT", "BARRIER", "STEP.COMMIT"}
        and ins.args["event"] in queues.waited_events
    }
    epoch: list[int] = []
    last_commit: int | None = None
    commit_before: list[int | None] = [None] * len(program)
    for pc, ins in enumerate(program):
        commit_before[pc] = last_commit
        if last_commit is not None:
            edge(last_commit, pc)
        if ins.op in {"WAIT", "BARRIER"}:
            for name in ins.args["events"]:
                if name not in event_pc:
                    raise ValueError(f"Unknown synchronization event {name}")
                edge(event_pc[name], pc)
        if ins.op == "STEP.COMMIT":
            for previous in epoch:
                edge(previous, pc)
            epoch = []
            last_commit = pc
        else:
            epoch.append(pc)
    return queues, tape_ids, owners, edges, indegree, commit_before


def _program_from_queues(queues: ProgramQueues) -> tuple[Instruction, ...]:
    commands: list[Instruction | None] = [None] * queues.instruction_count
    for tape in queues.groups:
        for command in tape.commands:
            if command is None:
                raise ValueError("Consumed workgroup queue")
            pc, ins = command
            commands[pc] = ins
    if any(ins is None for ins in commands):
        raise ValueError("Incomplete workgroup queue")
    return tuple(ins for ins in commands if ins is not None)


def validate_hbm_races(
    program: str | Iterable[Instruction] | None,
    *,
    replay: Callable[[], Iterable[Instruction]] | None = None,
    queues: ProgramQueues | None = None,
) -> None:
    """Raise ValueError for a cross-workgroup HBM RAW, WAR or WAW race.

    A source string or a replayable parsed stream can be scanned without
    materializing a one-workgroup program.  With a one-shot iterator, pass
    ``replay`` to regenerate it if a second workgroup is discovered.  Other
    instruction and memory-shape validation belongs to the functional machine.
    Pass ``program=None, queues=...`` to reuse unconsumed pipeline queues.
    """
    if queues is not None and sum(not tape.control for tape in queues.groups) <= 1:
        return
    if program is None:
        if queues is None:
            raise ValueError("Program or workgroup queues required")
        program = _program_from_queues(queues)
    if isinstance(program, str):
        source = program
        program = iter_parse(source)

        def replay():
            return iter_parse(source)

    if replay is not None:
        begins = 0
        for ins in program:
            begins += ins.op == "WG.BEGIN"
            if begins > 1:
                break
        if begins <= 1:
            return
        program = tuple(replay())
    else:
        program = tuple(program)
        if sum(ins.op == "WG.BEGIN" for ins in program) <= 1:
            return
    queues, tape_ids, owners, edges, indegree, commit_before = _graph(program, queues)
    ready = deque(pc for pc, degree in enumerate(indegree) if degree == 0)
    knowledge = [_EMPTY for _ in queues.groups]
    applied_commit: list[int | None] = [None] * len(queues.groups)
    sequence = [0] * len(queues.groups)
    # Only explicit WAIT/BARRIER operands need name lookup. HBM accesses and
    # local writes own their completion facts while they remain in an index;
    # retaining every event's snapshot made long programs quadratic in memory.
    event_knowledge: dict[str, _Knowledge] = {}
    wait_remaining = Counter(
        name
        for ins in program
        if ins.op in {"WAIT", "BARRIER"}
        for name in ins.args["events"]
    )
    commit_knowledge: dict[int, _Knowledge] = {}
    roots: list[_Node | None] = [None, None]  # reads, writes
    local_writes: dict[tuple[int, str, int], _Node | None] = {}
    visited = 0

    while ready:
        pc = ready.popleft()
        visited += 1
        ins = program[pc]
        op, args = ins.op, ins.args
        participants = owners[pc]
        preceding_commit = commit_before[pc]
        if preceding_commit is not None:
            for tape in participants:
                if applied_commit[tape] != preceding_commit:
                    knowledge[tape] = _join(
                        knowledge[tape], commit_knowledge[preceding_commit]
                    )
                    applied_commit[tape] = preceding_commit

        if op == "WG.BEGIN":
            tape = participants[0]
            predecessor = queues.groups[tape].predecessor
            if predecessor is not None:
                old = tape_ids[predecessor]
                knowledge[tape] = _join(
                    knowledge[tape], _completed(knowledge[old], old, sequence[old])
                )
        elif op in {"WAIT", "BARRIER"}:
            waited = _join(*(event_knowledge[name] for name in args["events"]))
            for name in args["events"]:
                wait_remaining[name] -= 1
                if not wait_remaining[name]:
                    del wait_remaining[name]
                    del event_knowledge[name]
            if op == "BARRIER":
                joined = _join(
                    waited,
                    *(_completed(knowledge[tape], tape, sequence[tape]) for tape in participants),
                )
                for tape in participants:
                    knowledge[tape] = joined
            else:
                tape = participants[0]
                knowledge[tape] = _join(knowledge[tape], waited)
        elif op == "STEP.COMMIT":
            joined = _join(
                *(
                    _completed(knowledge[tape], tape, sequence[tape])
                    for tape in range(len(queues.groups))
                )
            )
            commit_knowledge[pc] = joined
            for tape in participants:
                knowledge[tape] = joined
        elif op not in {"WG.END"}:
            tape = participants[0]
            sequence[tape] += 1
            identity = (tape, sequence[tape])
            reads, writes = _operands(ins)
            # The scheduler waits for a prior RF/SH write to finish before an
            # overlapping read or write can issue.  A prior read followed by
            # a write only waits for that read phase, so it is excluded here.
            seen_local: set[int] = set()
            for operand in (*reads, *writes):
                key = _local_key(tape, operand)
                if key is None:
                    continue
                lo, hi = _bounds(operand)
                for old in _candidates(local_writes.get(key), lo, hi):
                    if old.pc in seen_local or not _overlap(old.descriptor, operand):
                        continue
                    seen_local.add(old.pc)
                    knowledge[tape] = _join(knowledge[tape], old.completion)
            access: _Access | None = None
            if op in {"LD", "ST"}:
                descriptor = args["src"] if args["src"]["space"] == "HBM" else args["dst"]
                if descriptor["space"] == "HBM":
                    write = args["dst"]["space"] == "HBM"
                    lo, hi = _bounds(descriptor)
                    access = _Access(pc, tape, identity, descriptor, write, lo, hi)
                    old_accesses = list(_candidates(roots[1], lo, hi))
                    if write:
                        old_accesses.extend(_candidates(roots[0], lo, hi))
                    removable: list[_Access] = []
                    for old in old_accesses:
                        if not _overlap(old.descriptor, descriptor):
                            continue
                        if old.tape == tape:
                            # A conflicting transfer in the same tape cannot issue
                            # until the older transfer has completed.
                            assert old.completion is not None
                            knowledge[tape] = _join(
                                knowledge[tape],
                                old.completion,
                            )
                        elif not knowledge[tape].contains(old.identity):
                            first, second = (old, access) if old.pc < pc else (access, old)
                            kind = ("W" if first.write else "R") + ("W" if second.write else "R")
                            raise ValueError(
                                f"HBM {kind} race between lines "
                                f"{program[first.pc].line} and {program[second.pc].line}"
                            )
                        if (
                            old.descriptor == descriptor
                            and (write or not old.write)
                            and knowledge[tape].contains(old.identity)
                        ):
                            removable.append(old)
                    for old in removable:
                        index = int(old.write)
                        roots[index] = _erase(roots[index], _key(old))
                    index = int(write)
                    roots[index] = _insert(roots[index], _Node(access, _priority(pc)))
            completion = _event(knowledge[tape], identity)
            if args["event"] in queues.waited_events:
                event_knowledge[args["event"]] = completion
            if access is not None:
                access.completion = completion
            for operand in writes:
                key = _local_key(tape, operand)
                if key is None:
                    continue
                lo, hi = _bounds(operand)
                prior = list(_candidates(local_writes.get(key), lo, hi))
                for old in prior:
                    if old.descriptor == operand:
                        local_writes[key] = _erase(local_writes[key], _key(old))
                local_writes[key] = _insert(
                    local_writes.get(key),
                    _Node(
                        _LocalWrite(pc, operand, lo, hi, completion),
                        _priority(pc),
                    ),
                )

        for successor in edges[pc]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(program):
        raise ValueError("Cyclic synchronization dependency")
