"""FP32 functional execution for challenge ISA microprograms.

SparseHBM keeps the public 2 GiB address space without allocating untouched
pages. Functional execution still checks every actual read and write.
"""

from dataclasses import dataclass, field
from math import gcd
from tempfile import TemporaryFile

import numpy as np

from .address_views import dimensions
from .hardware import Hardware
from .isa import Instruction, sync_targets


def _f32(value):
    return np.asarray(value, dtype=np.float32)


def _overlapping_2d_write(shape: list[int], strides: list[int]) -> bool:
    """A repeated address exists iff the smallest stride-cancelling step fits."""
    stride_gcd = gcd(*strides)
    return shape[0] > strides[1] // stride_gcd and shape[1] > strides[0] // stride_gcd


def _tree_sum(values: np.ndarray) -> np.float32:
    level = list(_f32(values).ravel())
    if not level:
        raise ValueError("Empty reduction")
    while len(level) > 1:
        level = [
            np.float32(level[i] + level[i + 1]) if i + 1 < len(level) else level[i]
            for i in range(0, len(level), 2)
        ]
    return np.float32(level[0])


class SparseHBM:
    """Flat FP32 HBM backed by a sparse temporary file.

    The fixture loader writes ordinary slices. Untouched addresses read as
    zero, while MicroMachine separately enforces invalid scratch/output ranges.
    A MicroMachine takes ownership of this mutable object during execution.
    """

    dtype = np.dtype(np.float32)
    ndim = 1

    def __init__(self, words: int):
        if type(words) is not int or words <= 0 or words > 2**29:
            raise ValueError("Invalid sparse HBM size")
        self.size = words
        self._file = TemporaryFile()
        self._file.truncate(words * 4)
        self._data = np.memmap(self._file, dtype=np.float32, mode="r+", shape=(words,))
        self._valid_file = TemporaryFile()
        self._valid_file.truncate(words)
        self._valid = np.memmap(self._valid_file, dtype=np.uint8, mode="r+", shape=(words,))

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value


@dataclass
class Group:
    sm: int
    shared_quota: int
    rf: np.ndarray = field(default_factory=lambda: np.zeros(65536 // 4, dtype=np.float32))
    rf_valid: np.ndarray = field(default_factory=lambda: np.zeros(65536 // 4, dtype=bool))
    shared: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shared_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


class MicroMachine:
    def __init__(
        self,
        hardware: Hardware,
        hbm: np.ndarray | SparseHBM,
        readonly: list[tuple[int, int]] = None,
        unique_writes: list[tuple[int, int]] = None,
        release_rules: list[tuple[int, int, int]] = None,
        step_requirements: dict[int, list[tuple[int, int]]] = None,
        invalid_hbm: list[tuple[int, int]] = None,
    ):
        hardware.validate()
        if hbm.dtype != np.float32 or hbm.ndim != 1:
            raise ValueError("HBM must be a flat FP32 array")
        self.hardware = hardware
        self.hbm = hbm if isinstance(hbm, SparseHBM) else hbm.copy()
        self.invalid_hbm = list(invalid_hbm or [])
        self.valid = hbm._valid if isinstance(hbm, SparseHBM) else np.ones(hbm.size, dtype=bool)
        for lo, hi in self.invalid_hbm:
            if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= hbm.size:
                raise ValueError("Invalid HBM validity range")
            if not isinstance(hbm, SparseHBM):
                self.valid[lo:hi] = False
        self.readonly = readonly or []
        self.unique_writes = unique_writes or []
        self.release_rules = release_rules or []
        self.step_requirements = step_requirements or {}
        self.committed_steps = 0
        self.groups: dict[str, Group] = {}
        self.events: set[str] = set()
        self.written_hbm: set[int] = set()

    def _view(self, desc: dict, write: bool = False) -> tuple[np.ndarray, np.ndarray]:
        required = {"space", "offset", "count", "wg", "lane"}
        if not isinstance(desc, dict) or not required <= set(desc) <= required | {
            "shape",
            "strides",
        }:
            raise ValueError("Invalid memory operand")
        space, offset, count = desc["space"], desc["offset"], desc["count"]
        if space not in ("HBM", "RF", "SH"):
            raise ValueError("Invalid memory space")
        if type(offset) is not int or type(count) is not int or offset < 0 or count <= 0:
            raise ValueError("Invalid memory offset/count")
        if type(desc["lane"]) is not int or desc["lane"] < 0:
            raise ValueError("Invalid lane")
        if desc["wg"] is not None and (not isinstance(desc["wg"], str) or not desc["wg"]):
            raise ValueError("Invalid workgroup operand")
        shape, strides = dimensions(desc)
        if write and len(shape) == 2 and _overlapping_2d_write(shape, strides):
            raise ValueError("Overlapping write view")
        span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))

        def shaped(data, valid):
            if len(shape) == 1 and strides == [1]:
                return data[:count], valid[:count]
            byte_strides = tuple(stride * 4 for stride in strides)
            return (
                np.lib.stride_tricks.as_strided(data, shape=shape, strides=byte_strides),
                np.lib.stride_tricks.as_strided(valid, shape=shape, strides=tuple(strides)),
            )

        if space == "HBM":
            if desc["wg"] is not None or desc["lane"] != 0 or offset + span > self.hbm.size:
                raise ValueError("HBM address/scope invalid")
            if not write:
                for lo, hi, step in self.release_rules:
                    if offset < hi and offset + span > lo and self.committed_steps < step:
                        if any(lo <= i < hi for i in self._positions(desc)):
                            raise ValueError("HBM input not released")
            if write:
                for lo, hi in self.readonly:
                    if offset >= hi or offset + span <= lo:
                        continue
                    if any(lo <= i < hi for i in self._positions(desc)):
                        raise ValueError("Write to read-only HBM")
                for lo, hi in self.unique_writes:
                    if offset >= hi or offset + span <= lo:
                        continue
                    if any(
                        lo <= i < hi and i in self.written_hbm
                        for i in self._positions(desc)
                    ):
                        raise ValueError("Duplicate HBM output write")
            return shaped(self.hbm[offset : offset + span], self.valid[offset : offset + span])
        group = self.groups.get(desc["wg"])
        if group is None:
            raise ValueError("Unknown or ended workgroup")
        if space == "SH":
            if desc["lane"] != 0 or offset + span > group.shared.size:
                raise ValueError("Shared address/scope invalid")
            return shaped(
                group.shared[offset : offset + span], group.shared_valid[offset : offset + span]
            )
        lanes = self.hardware.vector_lanes
        words_per_lane = group.rf.size // lanes
        lane = desc["lane"]
        if lane >= lanes or offset + span > words_per_lane:
            raise ValueError("RF address/scope invalid")
        lo = lane * words_per_lane + offset
        return shaped(group.rf[lo : lo + span], group.rf_valid[lo : lo + span])

    @staticmethod
    def _positions(desc):
        shape = desc.get("shape", [desc["count"]])
        strides = desc.get("strides", [1])
        if len(shape) == 1:
            for i in range(shape[0]):
                yield desc["offset"] + i * strides[0]
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    yield desc["offset"] + i * strides[0] + j * strides[1]

    def _read(self, desc: dict) -> np.ndarray:
        data, valid = self._view(desc)
        if isinstance(self.hbm, SparseHBM) and desc["space"] == "HBM":
            offset = desc["offset"]
            shape = desc.get("shape", [desc["count"]])
            strides = desc.get("strides", [1])
            end = offset + 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
            if any(offset >= lo and end <= hi for lo, hi in self.invalid_hbm):
                if not valid.all():
                    raise ValueError("Read before write")
            elif any(offset < hi and end > lo for lo, hi in self.invalid_hbm):
                if any(
                    lo <= index < hi and not self.valid[index]
                    for index in self._positions(desc)
                    for lo, hi in self.invalid_hbm
                ):
                    raise ValueError("Read before write")
        elif not valid.all():
            raise ValueError("Read before write")
        return data.ravel().copy()

    def _write(self, desc: dict, values) -> None:
        data, valid = self._view(desc, write=True)
        val = _f32(values).ravel()
        if val.size != data.size or not np.isfinite(val).all():
            raise ValueError("Write shape or finite-value violation")
        data[...] = val.reshape(data.shape)
        valid[...] = True
        if desc["space"] == "HBM":
            self.written_hbm.update(self._positions(desc))

    def execute(
        self, program: tuple[Instruction, ...], *, hbm_races_validated: bool = False
    ) -> np.ndarray | SparseHBM:
        if not hbm_races_validated:
            from .hbm_race import validate_hbm_races

            program = tuple(program)
            validate_hbm_races(program)
        for instruction in program:
            op, args = instruction.op, instruction.args
            if op == "WG.BEGIN":
                if set(args) != {"wg", "sm", "shared_bytes"}:
                    raise ValueError("Invalid WG.BEGIN")
                name, sm, quota = args["wg"], args["sm"], args["shared_bytes"]
                if (
                    not isinstance(name, str)
                    or not name
                    or name in self.groups
                    or type(sm) is not int
                    or not 0 <= sm < self.hardware.sm_count
                    or type(quota) is not int
                    or quota < 0
                    or quota % 4
                ):
                    raise ValueError("Invalid workgroup")
                if quota > self.hardware.shared_kib * 1024:
                    raise ValueError("Workgroup shared quota exceeds SM capacity")
                self.groups[name] = Group(
                    sm=sm,
                    shared_quota=quota,
                    shared=np.zeros(quota // 4, np.float32),
                    shared_valid=np.zeros(quota // 4, bool),
                )
            elif op == "WG.END":
                if (
                    set(args) != {"wg"}
                    or not isinstance(args["wg"], str)
                    or args["wg"] not in self.groups
                ):
                    raise ValueError("Invalid WG.END")
                del self.groups[args["wg"]]
            elif op in ("LD", "ST"):
                if set(args) != {"src", "dst", "event"}:
                    raise ValueError("Invalid transfer")
                src, dst = args["src"], args["dst"]
                if not all(isinstance(operand, dict) for operand in (src, dst)):
                    raise ValueError("Invalid transfer operand")
                if not all({"space", "count", "wg"} <= set(operand) for operand in (src, dst)):
                    raise ValueError("Invalid transfer operand")
                if src["space"] == dst["space"] or src["count"] != dst["count"]:
                    raise ValueError("Invalid transfer spaces or length")
                if src["space"] != "HBM" and dst["space"] != "HBM":
                    if src["wg"] != dst["wg"]:
                        raise ValueError("Cross-workgroup transfer")
                self._write(dst, self._read(src))
                self._event(args["event"])
            elif op == "MMA.ACC":
                self._mma(args)
            elif op == "VEC":
                self._vec(args)
            elif op == "REDUCE":
                self._reduce(args)
            elif op == "SFU":
                self._sfu(args)
            elif op in ("WAIT", "BARRIER"):
                sync_targets(op, args, self.groups)
                if any(e not in self.events for e in args["events"]):
                    raise ValueError("Wait for unknown event")
            elif op == "STEP.COMMIT":
                if (
                    set(args) != {"step"}
                    or type(args["step"]) is not int
                    or args["step"] != self.committed_steps
                ):
                    raise ValueError("Invalid step commitment")
                requirements = self.step_requirements.get(self.committed_steps)
                if requirements is None or any(
                    index not in self.written_hbm
                    for lo, hi in requirements
                    for index in range(lo, hi)
                ):
                    raise ValueError("Step outputs incomplete")
                self.committed_steps += 1
            else:
                raise ValueError("Unknown operation")
        if self.groups:
            raise ValueError("Unclosed workgroups")
        if self.committed_steps != len(self.step_requirements):
            raise ValueError("Required step commitments incomplete")
        return self.hbm if isinstance(self.hbm, SparseHBM) else self.hbm.copy()

    def _event(self, name):
        if not isinstance(name, str) or not name or name in self.events:
            raise ValueError("Invalid or duplicate event")
        self.events.add(name)

    def _mma(self, args):
        if set(args) != {"a", "b", "acc", "m", "n", "k", "event"}:
            raise ValueError("Invalid MMA.ACC")
        if self.hardware.tc_count == 0:
            raise ValueError("MMA.ACC unavailable when TC=0")
        m, n, k = (args[x] for x in ("m", "n", "k"))
        if any(type(x) is not int or x <= 0 for x in (m, n, k)):
            raise ValueError("Invalid MMA shape")
        if any(not isinstance(args[x], dict) for x in ("a", "b", "acc")):
            raise ValueError("Invalid MMA operand")
        if any(not {"space", "wg"} <= set(args[x]) for x in ("a", "b", "acc")):
            raise ValueError("Invalid MMA operand")
        if any(args[x]["space"] != "RF" for x in ("a", "b", "acc")):
            raise ValueError("MMA operands must be RF")
        if len({args[x]["wg"] for x in ("a", "b", "acc")}) != 1:
            raise ValueError("Cross-workgroup MMA")
        a, b, acc = (self._read(args[x]) for x in ("a", "b", "acc"))
        if (a.size, b.size, acc.size) != (m * k, k * n, m * n):
            raise ValueError("MMA operand sizes mismatch")
        a, b, acc = a.reshape(m, k), b.reshape(k, n), acc.reshape(m, n)
        if k >= 16:
            parallel = self.hardware.tc_k_parallel
            for lo in range(0, k, parallel):
                terms = [
                    np.float32(a[:, lo + t, None] * b[None, lo + t, :])
                    if lo + t < k
                    else np.zeros((m, n), np.float32)
                    for t in range(parallel)
                ]
                while len(terms) > 1:
                    terms = [
                        np.float32(terms[i] + terms[i + 1]) if i + 1 < len(terms) else terms[i]
                        for i in range(0, len(terms), 2)
                    ]
                acc = np.float32(acc + terms[0])
            self._write(args["acc"], acc)
            self._event(args["event"])
            return
        parallel = self.hardware.tc_k_parallel
        for i in range(m):
            for j in range(n):
                value = acc[i, j]
                for lo in range(0, k, parallel):
                    terms = np.zeros(parallel, dtype=np.float32)
                    for t in range(parallel):
                        if lo + t < k:
                            terms[t] = np.float32(a[i, lo + t] * b[lo + t, j])
                    value = np.float32(value + _tree_sum(terms))
                acc[i, j] = value
        self._write(args["acc"], acc)
        self._event(args["event"])

    def _vec(self, args):
        if set(args) != {"kind", "src", "dst", "event"} or not isinstance(args["src"], list):
            raise ValueError("Invalid VEC")
        if not isinstance(args["dst"], dict) or any(not isinstance(x, dict) for x in args["src"]):
            raise ValueError("Invalid VEC operand")
        if not {"space", "wg", "count"} <= set(args["dst"]) or any(
            "space" in x and not {"wg", "count"} <= set(x) for x in args["src"]
        ):
            raise ValueError("Invalid VEC operand")
        if args["dst"]["space"] != "RF" or any(
            x.get("space") != "RF" and set(x) != {"imm"} for x in args["src"]
        ):
            raise ValueError("VEC operands must be RF")
        if len({x["wg"] for x in args["src"] if "space" in x} | {args["dst"]["wg"]}) != 1:
            raise ValueError("Cross-workgroup VEC")
        dst_count = args["dst"]["count"]
        values = []
        for operand in args["src"]:
            if "imm" in operand:
                value = operand["imm"]
                if type(value) not in (int, float):
                    raise ValueError("Nonfinite VEC immediate")
                try:
                    with np.errstate(over="ignore"):
                        rounded = np.float32(value)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError("Nonfinite VEC immediate") from exc
                if not np.isfinite(rounded):
                    raise ValueError("Nonfinite VEC immediate")
                values.append(np.full(dst_count, rounded, dtype=np.float32))
            else:
                value = self._read(operand)
                if value.size not in (1, dst_count):
                    raise ValueError("VEC size mismatch")
                values.append(value)
        kind = args["kind"]
        arity = {"add": 2, "sub": 2, "mul": 2, "div": 2, "fma": 3, "gt": 2, "select": 3, "max": 2}
        if kind not in arity or len(values) != arity[kind]:
            raise ValueError("Invalid VEC kind/arity")
        a, b = values[:2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = {
                "add": lambda: a + b,
                "sub": lambda: a - b,
                "mul": lambda: a * b,
                "div": lambda: a / b,
                "fma": lambda: _f32(a.astype(np.float64) * b + values[2]),
                "gt": lambda: (a > b).astype(np.float32),
                "select": lambda: np.where(a != 0, b, values[2]),
                "max": lambda: np.maximum(a, b),
            }[kind]()
        self._write(args["dst"], result)
        self._event(args["event"])

    def _reduce(self, args):
        if set(args) != {"kind", "src", "dst", "event"} or args["kind"] not in (
            "sum",
            "max",
        ):
            raise ValueError("Invalid REDUCE")
        if not isinstance(args["src"], dict) or not isinstance(args["dst"], dict):
            raise ValueError("Invalid REDUCE operand")
        if not all({"space", "wg"} <= set(args[x]) for x in ("src", "dst")):
            raise ValueError("Invalid REDUCE operand")
        if args["src"]["space"] != "RF" or args["dst"]["space"] != "RF":
            raise ValueError("REDUCE operands must be RF")
        if args["src"]["wg"] != args["dst"]["wg"]:
            raise ValueError("Cross-workgroup REDUCE")
        values = self._read(args["src"])
        result = _tree_sum(values) if args["kind"] == "sum" else np.max(values)
        self._write(args["dst"], [result])
        self._event(args["event"])

    def _sfu(self, args):
        if set(args) != {"kind", "src", "dst", "event"} or args["kind"] not in (
            "exp",
            "rsqrt",
            "tanh",
        ):
            raise ValueError("Invalid SFU")
        if not isinstance(args["src"], dict) or not isinstance(args["dst"], dict):
            raise ValueError("Invalid SFU operand")
        if not all({"space", "wg"} <= set(args[x]) for x in ("src", "dst")):
            raise ValueError("Invalid SFU operand")
        if args["src"]["space"] != "RF" or args["dst"]["space"] != "RF":
            raise ValueError("SFU operands must be RF")
        if args["src"]["wg"] != args["dst"]["wg"]:
            raise ValueError("Cross-workgroup SFU")
        values = self._read(args["src"])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = {
                "exp": lambda: np.exp(values),
                "rsqrt": lambda: 1 / np.sqrt(values),
                "tanh": lambda: np.tanh(values),
            }[args["kind"]]()
        self._write(args["dst"], result)
        self._event(args["event"])
