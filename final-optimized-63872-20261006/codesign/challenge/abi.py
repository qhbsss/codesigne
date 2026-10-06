"""Deterministic 2 GiB HBM symbol layout for the active challenge scenarios."""

from dataclasses import dataclass
from math import prod

from .workload import SCENARIOS, Model

HBM_BYTES = 2 * 1024**3
ALIGNMENT = 64


@dataclass(frozen=True)
class Symbol:
    name: str
    address: int
    shape: tuple[int, ...]
    readonly: bool
    release_step: int

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * 4


class Layout:
    def __init__(self, symbols: list[Symbol]):
        self.symbols = {item.name: item for item in symbols}
        if len(self.symbols) != len(symbols):
            raise ValueError("Duplicate ABI symbol")
        self.total_bytes = max((s.address + s.nbytes for s in symbols), default=0)
        if self.total_bytes > HBM_BYTES:
            raise ValueError("ABI exceeds 2 GiB HBM")

    def address(
        self, name: str, index: tuple[int, ...], committed_steps: int, write: bool = False
    ) -> int:
        if name not in self.symbols:
            raise ValueError(f"Unknown ABI symbol: {name}")
        symbol = self.symbols[name]
        if write and symbol.readonly:
            raise ValueError("Write to read-only ABI symbol")
        if type(committed_steps) is not int or committed_steps < symbol.release_step:
            raise ValueError("Input not released yet")
        if len(index) != len(symbol.shape) or any(
            type(i) is not int or not 0 <= i < n for i, n in zip(index, symbol.shape)
        ):
            raise ValueError("ABI index out of bounds")
        offset = 0
        for i, extent in zip(index, symbol.shape):
            offset = offset * extent + i
        return symbol.address + offset * 4


def build_layout(model: Model, scenario: str) -> Layout:
    if scenario not in SCENARIOS or model.width % model.heads:
        raise ValueError("Unknown scenario or invalid model")
    batch, context, new_positions = SCENARIOS[scenario]
    prefill = scenario.startswith("P")
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    symbols = []
    cursor = 0

    def add(name: str, shape: tuple[int, ...], readonly: bool, release_step: int = 0):
        nonlocal cursor
        cursor = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        symbols.append(Symbol(name, cursor, shape, readonly, release_step))
        cursor += prod(shape) * 4

    for layer in range(model.layers):
        prefix = f"layer{layer}"
        for name, shape in (
            ("ln1_g", (d,)),
            ("ln1_b", (d,)),
            ("wqkv", (d, 3 * d)),
            ("wo", (d, d)),
            ("ln2_g", (d,)),
            ("ln2_b", (d,)),
            ("w1", (d, f)),
            ("b1", (f,)),
            ("w2", (f, d)),
            ("b2", (d,)),
        ):
            add(f"{prefix}/{name}", shape, True)
        if not prefill:
            add(f"{prefix}/history_k", (batch, h, context, hd), True)
            add(f"{prefix}/history_v", (batch, h, context, hd), True)
    if prefill:
        add("input/prompt", (batch, context, d), True)
        add("input/step0", (batch, d), True, 1)
        add("output/hidden", (batch, context + 1, d), False)
        kv_steps = context + 1
    else:
        for step in range(new_positions):
            add(f"input/step{step}", (batch, d), True, step)
        add("output/hidden", (batch, new_positions, d), False)
        kv_steps = new_positions
    for layer in range(model.layers):
        for kind in ("k", "v"):
            add(f"layer{layer}/new_{kind}", (batch, h, kv_steps, hd), False)
    scratch_start = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    add("scratch", ((HBM_BYTES - scratch_start) // 4,), False)
    return Layout(symbols)
