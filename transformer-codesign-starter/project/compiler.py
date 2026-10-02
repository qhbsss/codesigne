"""Generate literal-ISA baselines for the two active trial cases.

The Python here is a compiler only.  The emitted program uses no operator-level
instructions for LayerNorm, attention, softmax, or GELU.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt

from codesign.challenge.abi import Layout, build_layout
from codesign.challenge.workload import MODELS, SCENARIOS


def var(name):
    return {"var": name}


def add(*items):
    return {"add": list(items)}


def mul(*items):
    return {"mul": list(items)}


def minimum(*items):
    return {"min": list(items)}


def sub(a, b):
    return add(a, mul(-1, b))


def hbm(offset, count, shape=None, strides=None):
    result = {"space": "HBM", "offset": offset, "count": count, "wg": None, "lane": 0}
    if shape is not None:
        result.update(shape=shape, strides=strides)
    return result


def rf(lane, count, offset=0, wg="g"):
    return {"space": "RF", "offset": offset, "count": count, "wg": wg, "lane": lane}


def imm(value):
    return {"imm": value}


@dataclass(frozen=True)
class Tensor:
    offset: int
    shape: tuple[int, ...]


class Scratch:
    def __init__(self, base: int):
        self.cursor = base

    def alloc(self, *shape: int) -> Tensor:
        size = 1
        for extent in shape:
            size *= extent
        result = Tensor(self.cursor, tuple(shape))
        self.cursor += size
        return result


def columns(width: int, parts: int, index: int) -> tuple[int, int]:
    if width % parts:
        raise ValueError(f"{width} is not divisible by {parts}")
    chunk = width // parts
    return index * chunk, (index + 1) * chunk


def heads_for(heads: int, parts: int, index: int) -> tuple[int, int] | None:
    """Head slice owned by this worker. Extra workers may own none."""
    if heads % parts == 0:
        chunk = heads // parts
        return index * chunk, (index + 1) * chunk
    if parts % heads == 0:
        return (index, index + 1) if index < heads else None
    raise ValueError(f"{heads} heads cannot be split across {parts} workers")


def emit_barrier(lines: list[str], workers: list["Builder"]) -> None:
    payload = {"wgs": [worker.wg for worker in workers], "events": []}
    lines.append("BARRIER " + json.dumps(payload, separators=(",", ":")))


class Builder:
    def __init__(self, layout: Layout, wg: str = "g", sm: int = 0, memory: Scratch | None = None, lines: list[str] | None = None, gemm_n_tile: int = 16):
        if gemm_n_tile not in (8, 16):
            raise ValueError("Unsupported GEMM N tile")
        self.layout = layout
        self.wg = wg
        self.gemm_n_tile = gemm_n_tile
        self.lines = [] if lines is None else lines
        self.loops = []
        self.serial = 0
        self.memory = Scratch(layout.symbols["scratch"].address // 4) if memory is None else memory
        self.tensors = {}
        self.emit("WG.BEGIN", wg=wg, sm=sm, shared_bytes=0)

    def symbol(self, name):
        item = self.layout.symbols[name]
        return Tensor(item.address // 4, item.shape)

    def alloc(self, name, *shape):
        result = self.memory.alloc(*shape)
        self.tensors[name] = result
        return result

    def _rf(self, lane, count, offset=0):
        return rf(lane, count, offset, wg=self.wg)

    def emit(self, op, **args):
        if "event" in args and args["event"] is None:
            self.serial += 1
            suffix = "".join(f"_{{{name}}}" for name in self.loops)
            args["event"] = f"{self.wg}_e{self.serial}{suffix}"
        self.lines.append(f"{op} {json.dumps(args, separators=(',', ':'))}")

    @contextmanager
    def loop(self, name, start, stop, step=1):
        ident = f"{self.wg}_{name}"
        self.emit("FOR", var=ident, start=start, stop=stop, step=step)
        self.loops.append(ident)
        yield var(ident)
        self.loops.pop()
        self.emit("END.FOR")

    def ld(self, source, target):
        self.emit("LD", src=source, dst=target, event=None)

    def st(self, source, target):
        self.emit("ST", src=source, dst=target, event=None)

    def vec(self, kind, sources, target):
        self.emit("VEC", kind=kind, src=sources, dst=target, event=None)

    def reduce(self, kind, source, target):
        self.emit("REDUCE", kind=kind, src=source, dst=target, event=None)

    def sfu(self, kind, source, target):
        self.emit("SFU", kind=kind, src=source, dst=target, event=None)

    def _lane_pair(self, index, base: int = 1, span: int = 5):
        """Even chunks use `base`, odd chunks use `base + span` (lanes 1 and 6)."""
        if isinstance(index, int):
            return base + (index % 2) * span
        return add(base, mul({"mod": [index, 2]}, span))

    def _pipeline_chunks(self, length: int, chunk: int, name: str, load, compute, origin: int = 0):
        """Prefetch the next chunk into the idle buffer while computing the current one.

        `load(lane, start, depth)` and `compute(lane, start, depth)` see chunk
        coordinates along an axis of `length` elements beginning at `origin`.
        Non-tail chunks are exactly `chunk` long, so K reduction order is unchanged.
        """
        if length <= 0 or chunk <= 0:
            raise ValueError("invalid pipeline")
        n_chunks = (length + chunk - 1) // chunk

        def position(index):
            scaled = index * chunk if isinstance(index, int) else mul(index, chunk)
            return scaled if origin == 0 else add(origin, scaled)

        first_depth = min(chunk, length)
        load(self._lane_pair(0), position(0), first_depth)
        if n_chunks == 1:
            compute(self._lane_pair(0), position(0), first_depth)
            return
        with self.loop(f"{name}c", 0, n_chunks - 1) as index:
            nxt = add(index, 1)
            load(
                self._lane_pair(nxt),
                position(nxt),
                minimum(chunk, sub(length, mul(nxt, chunk))),
            )
            compute(self._lane_pair(index), position(index), chunk)
        last = n_chunks - 1
        compute(self._lane_pair(last), position(last), length - last * chunk)

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        out: Tensor,
        m_extent: int,
        k_extent: int,
        n_extent: int,
        name: str,
        n_lo: int = 0,
        n_hi: int | None = None,
        epilogue: str | None = None,
        bias: Tensor | None = None,
        residual: Tensor | None = None,
    ):
        if n_hi is None:
            n_hi = n_extent
        if b.shape != (k_extent, n_extent):
            raise ValueError("GEMM weight shape")
        tile_m = 8
        tile_n = self.gemm_n_tile
        if n_lo < 0 or n_hi > n_extent or (n_hi - n_lo) % tile_n:
            raise ValueError("GEMM N tile range")
        if epilogue not in (None, "residual", "bias_gelu", "residual_bias"):
            raise ValueError("Unsupported GEMM epilogue")
        if epilogue in ("bias_gelu", "residual_bias") and bias is None:
            raise ValueError("Fused bias is missing")
        if epilogue in ("residual", "residual_bias") and residual is None:
            raise ValueError("Fused residual is missing")
        # One weight tile stays in RF and is reused across every row tile.
        if ((m_extent + tile_m - 1) // tile_m) * tile_m * tile_n > 1024:
            raise ValueError("GEMM accumulators exceed a register lane")
        if 48 * tile_n > 1024:
            raise ValueError("GEMM weight tile exceeds a register lane")
        with self.loop(f"{name}n", n_lo, n_hi, tile_n) as col:
            with self.loop(f"{name}z", 0, m_extent, tile_m) as row:
                rows = minimum(tile_m, sub(m_extent, row))
                cells = mul(rows, tile_n)
                self.vec("add", [imm(0), imm(0)], self._rf(2, cells, mul(row, tile_n)))

            def load_b(lane, start, depth):
                self.ld(
                    hbm(
                        add(b.offset, mul(start, n_extent), col),
                        mul(depth, tile_n),
                        [depth, tile_n],
                        [n_extent, 1],
                    ),
                    self._rf(lane, mul(depth, tile_n)),
                )

            def use_b(lane, start, depth):
                with self.loop(f"{name}m", 0, m_extent, tile_m) as row:
                    rows = minimum(tile_m, sub(m_extent, row))
                    self.ld(
                        hbm(
                            add(a.offset, mul(row, k_extent), start),
                            mul(rows, depth),
                            [rows, depth],
                            [k_extent, 1],
                        ),
                        self._rf(0, mul(rows, depth)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(0, mul(rows, depth)),
                        b=self._rf(lane, mul(depth, tile_n)),
                        acc=self._rf(2, mul(rows, tile_n), mul(row, tile_n)),
                        m=rows,
                        n=tile_n,
                        k=depth,
                        event=None,
                    )

            self._pipeline_chunks(k_extent, 48, f"{name}k", load_b, use_b)
            if epilogue is None:
                with self.loop(f"{name}s", 0, m_extent, tile_m) as row:
                    rows = minimum(tile_m, sub(m_extent, row))
                    cells = mul(rows, tile_n)
                    self.st(
                        self._rf(2, cells, mul(row, tile_n)),
                        hbm(
                            add(out.offset, mul(row, n_extent), col),
                            cells,
                            [rows, tile_n],
                            [n_extent, 1],
                        ),
                    )
            else:
                with self.loop(f"{name}s", 0, m_extent, tile_m) as row:
                    rows = minimum(tile_m, sub(m_extent, row))
                    with self.loop(f"{name}e", 0, rows) as inner:
                        row_index = add(row, inner)
                        self._epilogue_row(
                            out,
                            n_extent,
                            col,
                            row_index,
                            mul(row_index, tile_n),
                            tile_n,
                            epilogue,
                            bias,
                            residual,
                        )

    def _epilogue_row(
        self,
        out: Tensor,
        n_extent: int,
        col,
        row_index,
        acc_off,
        n: int,
        epilogue: str,
        bias: Tensor | None,
        residual: Tensor | None,
    ):
        """Fold bias, GELU, and residual into one register tile before its only store."""
        acc = self._rf(2, n, acc_off)
        if epilogue == "bias_gelu":
            self.ld(hbm(add(bias.offset, col), n), self._rf(3, n))
            self.vec("add", [acc, self._rf(3, n)], self._rf(5, n))
            self._gelu_rf(5, n)
            stored = self._rf(0, n)
        elif epilogue == "residual":
            self.ld(
                hbm(add(residual.offset, mul(row_index, n_extent), col), n),
                self._rf(0, n),
            )
            self.vec("add", [self._rf(0, n), acc], self._rf(5, n))
            stored = self._rf(5, n)
        else:
            self.ld(
                hbm(add(residual.offset, mul(row_index, n_extent), col), n),
                self._rf(0, n),
            )
            self.vec("add", [self._rf(0, n), acc], self._rf(5, n))
            self.ld(hbm(add(bias.offset, col), n), self._rf(3, n))
            self.vec("add", [self._rf(5, n), self._rf(3, n)], self._rf(0, n))
            stored = self._rf(0, n)
        self.st(stored, hbm(add(out.offset, mul(row_index, n_extent), col), n))

    def _gelu_rf(self, lane: int, n: int):
        """tanh-approximation GELU. `lane` holds x and is not overwritten; result is lane 0."""
        x = self._rf(lane, n)
        self.vec("mul", [x, x], self._rf(1, n))
        self.vec("mul", [self._rf(1, n), x], self._rf(1, n))
        self.vec("fma", [self._rf(1, n), imm(0.044715), x], self._rf(1, n))
        self.vec("mul", [self._rf(1, n), imm(sqrt(2 / 3.141592653589793))], self._rf(1, n))
        self.sfu("tanh", self._rf(1, n), self._rf(1, n))
        self.vec("add", [self._rf(1, n), imm(1)], self._rf(1, n))
        self.vec("mul", [x, imm(0.5)], self._rf(0, n))
        self.vec("mul", [self._rf(0, n), self._rf(1, n)], self._rf(0, n))

    def export_decode_kv(
        self,
        qkv: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        step: int,
        d: int,
        head: int,
        hd: int,
    ):
        for kind, target, component in (("k", new_k, 1), ("v", new_v, 2)):
            source = qkv.offset + component * d + head * hd
            destination = target.offset + head * target.shape[2] * hd + step * hd
            self.ld(hbm(source, hd), self._rf(0, hd))
            self.st(self._rf(0, hd), hbm(destination, hd))

    def score_kv_range(
        self,
        qkv: Tensor,
        source: Tensor,
        stride: int,
        score_base: int,
        head: int,
        key_lo: int,
        key_hi: int,
        index_base: int,
        hd: int,
        name: str,
    ):
        """Score one key interval of one head and spill that interval to HBM."""
        self.ld(hbm(qkv.offset + head * hd, hd), self._rf(0, hd))
        span = key_hi - key_lo

        def load_k(lane, key, n):
            self.ld(
                hbm(
                    add(source.offset, head * stride * hd, mul(key, hd)),
                    mul(hd, n),
                    [hd, n],
                    [1, hd],
                ),
                self._rf(lane, mul(hd, n)),
            )

        def use_k(lane, key, n):
            placed = key if index_base == 0 else add(index_base, key)
            self.vec("add", [imm(0), imm(0)], self._rf(4, n))
            self.emit(
                "MMA.ACC",
                a=self._rf(0, hd),
                b=self._rf(lane, mul(hd, n)),
                acc=self._rf(4, n),
                m=1,
                n=n,
                k=hd,
                event=None,
            )
            self.vec("mul", [self._rf(4, n), imm(1 / sqrt(hd))], self._rf(4, n))
            self.st(self._rf(4, n), hbm(add(score_base, placed), n))

        self._pipeline_chunks(span, 16, name, load_k, use_k, origin=key_lo)

    def softmax_scores(self, score_base: int, total: int):
        """Same full-vector softmax as the single-worker path, after scores are gathered."""
        self.ld(hbm(score_base, total), self._rf(4, total))
        self.reduce("max", self._rf(4, total), self._rf(1, 1))
        self.vec("sub", [self._rf(4, total), self._rf(1, 1)], self._rf(4, total))
        self.sfu("exp", self._rf(4, total), self._rf(4, total))
        self.reduce("sum", self._rf(4, total), self._rf(1, 1))
        self.vec("div", [self._rf(4, total), self._rf(1, 1)], self._rf(4, total))
        self.st(self._rf(4, total), hbm(score_base, total))

    def context_kv_range(
        self,
        source: Tensor,
        stride: int,
        score_base: int,
        dest: int,
        head: int,
        key_lo: int,
        key_hi: int,
        index_base: int,
        hd: int,
        name: str,
    ):
        self.vec("add", [imm(0), imm(0)], self._rf(2, hd))
        span = key_hi - key_lo

        def load_v(lane, key, n):
            self.ld(
                hbm(add(source.offset, head * stride * hd, mul(key, hd)), mul(n, hd)),
                self._rf(lane, mul(n, hd)),
            )

        def use_v(lane, key, n):
            placed = key if index_base == 0 else add(index_base, key)
            self.ld(hbm(add(score_base, placed), n), self._rf(4, n))
            self.emit(
                "MMA.ACC",
                a=self._rf(4, n),
                b=self._rf(lane, mul(n, hd)),
                acc=self._rf(2, hd),
                m=1,
                n=hd,
                k=n,
                event=None,
            )

        self._pipeline_chunks(span, 16, name, load_v, use_v, origin=key_lo)
        self.st(self._rf(2, hd), hbm(dest, hd))

    def combine_context(self, parts: list[int], dest: int, hd: int):
        """Add key-range partials in key order."""
        self.ld(hbm(parts[0], hd), self._rf(0, hd))
        for addr in parts[1:]:
            self.ld(hbm(addr, hd), self._rf(1, hd))
            self.vec("add", [self._rf(0, hd), self._rf(1, hd)], self._rf(2, hd))
            self.vec("add", [self._rf(2, hd), imm(0)], self._rf(0, hd))
        self.st(self._rf(0, hd), hbm(dest, hd))

    def layernorm(
        self, source: Tensor, gamma: Tensor, beta: Tensor, out: Tensor, rows: int, d: int, name: str
    ):
        with self.loop(name, 0, rows) as row:
            base = add(source.offset, mul(row, d))
            target = add(out.offset, mul(row, d))
            self.ld(hbm(base, d), self._rf(0, d))
            self.reduce("sum", self._rf(0, d), self._rf(1, 1))
            self.vec("mul", [self._rf(1, 1), imm(1 / d)], self._rf(1, 1))
            self.vec("sub", [self._rf(0, d), self._rf(1, 1)], self._rf(2, d))
            self.vec("mul", [self._rf(2, d), self._rf(2, d)], self._rf(3, d))
            self.reduce("sum", self._rf(3, d), self._rf(1, 1))
            self.vec("mul", [self._rf(1, 1), imm(1 / d)], self._rf(1, 1))
            self.vec("add", [self._rf(1, 1), imm(1e-5)], self._rf(1, 1))
            self.sfu("rsqrt", self._rf(1, 1), self._rf(1, 1))
            self.vec("mul", [self._rf(2, d), self._rf(1, 1)], self._rf(2, d))
            self.ld(hbm(gamma.offset, d), self._rf(3, d))
            self.vec("mul", [self._rf(2, d), self._rf(3, d)], self._rf(2, d))
            self.ld(hbm(beta.offset, d), self._rf(3, d))
            self.vec("add", [self._rf(2, d), self._rf(3, d)], self._rf(2, d))
            self.st(self._rf(2, d), hbm(target, d))

    def add_rows(
        self,
        left: Tensor,
        right: Tensor,
        out: Tensor,
        rows: int,
        width: int,
        name: str,
        bias: Tensor | None = None,
        col_lo: int = 0,
        col_hi: int | None = None,
    ):
        if col_hi is None:
            col_hi = width
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", col_lo, col_hi, 384) as col:
                n = minimum(384, sub(col_hi, col))
                self.ld(hbm(add(left.offset, mul(row, width), col), n), self._rf(0, n))
                self.ld(hbm(add(right.offset, mul(row, width), col), n), self._rf(1, n))
                self.vec("add", [self._rf(0, n), self._rf(1, n)], self._rf(2, n))
                if bias is not None:
                    self.ld(hbm(add(bias.offset, col), n), self._rf(3, n))
                    self.vec("add", [self._rf(2, n), self._rf(3, n)], self._rf(2, n))
                self.st(self._rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def add_bias(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str, col_lo: int = 0, col_hi: int | None = None):
        if col_hi is None:
            col_hi = width
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", col_lo, col_hi, 384) as col:
                n = minimum(384, sub(col_hi, col))
                self.ld(hbm(add(source.offset, mul(row, width), col), n), self._rf(0, n))
                self.ld(hbm(add(bias.offset, col), n), self._rf(1, n))
                self.vec("add", [self._rf(0, n), self._rf(1, n)], self._rf(2, n))
                self.st(self._rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def gelu(self, source: Tensor, out: Tensor, rows: int, width: int, name: str, col_lo: int = 0, col_hi: int | None = None):
        if col_hi is None:
            col_hi = width
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", col_lo, col_hi, 384) as col:
                n = minimum(384, sub(col_hi, col))
                addr = add(source.offset, mul(row, width), col)
                self.ld(hbm(addr, n), self._rf(0, n))
                self.vec("mul", [self._rf(0, n), self._rf(0, n)], self._rf(1, n))
                self.vec("mul", [self._rf(1, n), self._rf(0, n)], self._rf(1, n))
                self.vec("fma", [self._rf(1, n), imm(0.044715), self._rf(0, n)], self._rf(1, n))
                self.vec("mul", [self._rf(1, n), imm(sqrt(2 / 3.141592653589793))], self._rf(1, n))
                self.sfu("tanh", self._rf(1, n), self._rf(1, n))
                self.vec("add", [self._rf(1, n), imm(1)], self._rf(1, n))
                self.vec("mul", [self._rf(0, n), imm(0.5)], self._rf(0, n))
                self.vec("mul", [self._rf(0, n), self._rf(1, n)], self._rf(0, n))
                self.st(self._rf(0, n), hbm(add(out.offset, mul(row, width), col), n))

    def _attention_blocked(
        self,
        qkv: Tensor,
        context: Tensor,
        k_out: Tensor,
        v_out: Tensor,
        rows: int,
        d: int,
        hd: int,
        name: str,
        head_lo: int,
        head_hi: int,
    ):
        """Score a 16-row block against each full K/V tile once, then the causal tail."""
        stride = rows
        scale = 1 / sqrt(hd)
        seq = k_out.shape[2]
        for head in range(head_lo, head_hi):
            k_base = k_out.offset + head * seq * hd
            v_base = v_out.offset + head * v_out.shape[2] * hd
            with self.loop(f"{name}b{head}", 0, rows // 16) as block_i:
                block = mul(block_i, 16)
                with self.loop(f"{name}q{head}", 0, 16) as local:
                    row = add(block, local)
                    self.ld(
                        hbm(add(qkv.offset, mul(row, 3 * d), head * hd), hd),
                        self._rf(0, hd, mul(local, hd)),
                    )
                with self.loop(f"{name}t{head}", 0, block_i) as tile:
                    key = mul(tile, 16)
                    self.ld(
                        hbm(add(k_base, mul(key, hd)), hd * 16, [hd, 16], [1, hd]),
                        self._rf(1, hd * 16),
                    )
                    with self.loop(f"{name}r{head}", 0, 16) as local:
                        slot = add(mul(local, stride), key)
                        self.vec("add", [imm(0), imm(0)], self._rf(4, 16, slot))
                        self.emit(
                            "MMA.ACC",
                            a=self._rf(0, hd, mul(local, hd)),
                            b=self._rf(1, hd * 16),
                            acc=self._rf(4, 16, slot),
                            m=1,
                            n=16,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [self._rf(4, 16, slot), imm(scale)], self._rf(4, 16, slot))
                with self.loop(f"{name}p{head}", 0, 16) as local:
                    n = add(local, 1)
                    slot = add(mul(local, stride), block)
                    self.ld(
                        hbm(add(k_base, mul(block, hd)), mul(hd, n), [hd, n], [1, hd]),
                        self._rf(1, mul(hd, n)),
                    )
                    self.vec("add", [imm(0), imm(0)], self._rf(4, n, slot))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(0, hd, mul(local, hd)),
                        b=self._rf(1, mul(hd, n)),
                        acc=self._rf(4, n, slot),
                        m=1,
                        n=n,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [self._rf(4, n, slot), imm(scale)], self._rf(4, n, slot))
                with self.loop(f"{name}s{head}", 0, 16) as local:
                    limit = add(block, local, 1)
                    slot = mul(local, stride)
                    self.reduce("max", self._rf(4, limit, slot), self._rf(1, 1))
                    self.vec("sub", [self._rf(4, limit, slot), self._rf(1, 1)], self._rf(4, limit, slot))
                    self.sfu("exp", self._rf(4, limit, slot), self._rf(4, limit, slot))
                    self.reduce("sum", self._rf(4, limit, slot), self._rf(1, 1))
                    self.vec("div", [self._rf(4, limit, slot), self._rf(1, 1)], self._rf(4, limit, slot))
                with self.loop(f"{name}z{head}", 0, 16) as local:
                    self.vec("add", [imm(0), imm(0)], self._rf(2, hd, mul(local, hd)))
                with self.loop(f"{name}u{head}", 0, block_i) as tile:
                    key = mul(tile, 16)
                    self.ld(hbm(add(v_base, mul(key, hd)), 16 * hd), self._rf(1, 16 * hd))
                    with self.loop(f"{name}w{head}", 0, 16) as local:
                        self.emit(
                            "MMA.ACC",
                            a=self._rf(4, 16, add(mul(local, stride), key)),
                            b=self._rf(1, 16 * hd),
                            acc=self._rf(2, hd, mul(local, hd)),
                            m=1,
                            n=hd,
                            k=16,
                            event=None,
                        )
                with self.loop(f"{name}y{head}", 0, 16) as local:
                    n = add(local, 1)
                    row = add(block, local)
                    self.ld(hbm(add(v_base, mul(block, hd)), mul(n, hd)), self._rf(1, mul(n, hd)))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(4, n, add(mul(local, stride), block)),
                        b=self._rf(1, mul(n, hd)),
                        acc=self._rf(2, hd, mul(local, hd)),
                        m=1,
                        n=hd,
                        k=n,
                        event=None,
                    )
                    self.st(
                        self._rf(2, hd, mul(local, hd)),
                        hbm(add(context.offset, mul(row, d), head * hd), hd),
                    )

    def attention(
        self,
        qkv: Tensor,
        context: Tensor,
        k_out: Tensor,
        v_out: Tensor,
        rows: int,
        past: int,
        d: int,
        h: int,
        hd: int,
        name: str,
        head_lo: int = 0,
        head_hi: int | None = None,
    ):
        if head_hi is None:
            head_hi = h
        # Export every layer's K/V in the public head-major ABI.
        with self.loop(f"{name}h", head_lo, head_hi) as head:
            with self.loop(f"{name}r", 0, rows) as row:
                for kind, target, component in (("k", k_out, 1), ("v", v_out, 2)):
                    source_addr = add(qkv.offset, mul(row, 3 * d), component * d, mul(head, hd))
                    target_addr = add(
                        target.offset, mul(head, target.shape[2] * hd), mul(add(past, row), hd)
                    )
                    self.ld(hbm(source_addr, hd), self._rf(0, hd))
                    self.st(self._rf(0, hd), hbm(target_addr, hd))
        if past == 0 and rows % 16 == 0 and 0 < rows <= 64:
            self._attention_blocked(qkv, context, k_out, v_out, rows, d, hd, name, head_lo, head_hi)
            return
        with self.loop(f"{name}q", 0, rows) as row:
            limit = add(past, row, 1)
            with self.loop(f"{name}head", head_lo, head_hi) as head:
                q_addr = add(qkv.offset, mul(row, 3 * d), mul(head, hd))
                self.ld(hbm(q_addr, hd), self._rf(0, hd))
                with self.loop(f"{name}key", 0, limit, 16) as key:
                    n = minimum(16, sub(limit, key))
                    cells = mul(hd, n)
                    self.ld(
                        hbm(
                            add(k_out.offset, mul(head, k_out.shape[2] * hd), mul(key, hd)),
                            cells,
                            [hd, n],
                            [1, hd],
                        ),
                        self._rf(1, cells),
                    )
                    self.vec("add", [imm(0), imm(0)], self._rf(4, n, key))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(0, hd),
                        b=self._rf(1, cells),
                        acc=self._rf(4, n, key),
                        m=1,
                        n=n,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [self._rf(4, n, key), imm(1 / sqrt(hd))], self._rf(4, n, key))
                self.reduce("max", self._rf(4, limit), self._rf(1, 1))
                self.vec("sub", [self._rf(4, limit), self._rf(1, 1)], self._rf(4, limit))
                self.sfu("exp", self._rf(4, limit), self._rf(4, limit))
                self.reduce("sum", self._rf(4, limit), self._rf(1, 1))
                self.vec("div", [self._rf(4, limit), self._rf(1, 1)], self._rf(4, limit))
                self.vec("add", [imm(0), imm(0)], self._rf(2, hd))
                with self.loop(f"{name}value", 0, limit, 16) as key:
                    depth = minimum(16, sub(limit, key))
                    self.ld(
                        hbm(
                            add(v_out.offset, mul(head, v_out.shape[2] * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        self._rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(4, depth, key),
                        b=self._rf(1, mul(depth, hd)),
                        acc=self._rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
                self.st(self._rf(2, hd), hbm(add(context.offset, mul(row, d), mul(head, hd)), hd))

    def attention_decode(
        self,
        qkv: Tensor,
        context: Tensor,
        history_k: Tensor,
        history_v: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        step: int,
        past: int,
        d: int,
        h: int,
        hd: int,
        name: str,
        head_lo: int = 0,
        head_hi: int | None = None,
    ):
        if head_hi is None:
            head_hi = h
        """One decode step: score historical and newly generated KV separately."""
        generated = step + 1
        total = past + generated
        for kind, target, component in (("k", new_k, 1), ("v", new_v, 2)):
            with self.loop(f"{name}{kind}h", head_lo, head_hi) as head:
                source = add(qkv.offset, component * d, mul(head, hd))
                destination = add(target.offset, mul(head, target.shape[2] * hd), step * hd)
                self.ld(hbm(source, hd), self._rf(0, hd))
                self.st(self._rf(0, hd), hbm(destination, hd))
        with self.loop(f"{name}head", head_lo, head_hi) as head:
            self.ld(hbm(add(qkv.offset, mul(head, hd)), hd), self._rf(0, hd))
            for source, count, score_base, stride in (
                (history_k, past, 0, past),
                (new_k, generated, past, new_k.shape[2]),
            ):
                with self.loop(f"{name}key{score_base}", 0, count, 16) as key:
                    n = minimum(16, sub(count, key))
                    cells = mul(hd, n)
                    self.ld(
                        hbm(
                            add(source.offset, mul(head, stride * hd), mul(key, hd)),
                            cells,
                            [hd, n],
                            [1, hd],
                        ),
                        self._rf(1, cells),
                    )
                    placed = add(score_base, key)
                    self.vec("add", [imm(0), imm(0)], self._rf(4, n, placed))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(0, hd),
                        b=self._rf(1, cells),
                        acc=self._rf(4, n, placed),
                        m=1,
                        n=n,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [self._rf(4, n, placed), imm(1 / sqrt(hd))], self._rf(4, n, placed))
            self.reduce("max", self._rf(4, total), self._rf(1, 1))
            self.vec("sub", [self._rf(4, total), self._rf(1, 1)], self._rf(4, total))
            self.sfu("exp", self._rf(4, total), self._rf(4, total))
            self.reduce("sum", self._rf(4, total), self._rf(1, 1))
            self.vec("div", [self._rf(4, total), self._rf(1, 1)], self._rf(4, total))
            self.vec("add", [imm(0), imm(0)], self._rf(2, hd))
            for source, count, prob_base, stride in (
                (history_v, past, 0, past),
                (new_v, generated, past, new_v.shape[2]),
            ):
                with self.loop(f"{name}value{prob_base}", 0, count, 16) as key:
                    depth = minimum(16, sub(count, key))
                    placed = add(prob_base, key)
                    self.ld(
                        hbm(
                            add(source.offset, mul(head, stride * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        self._rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(4, depth, placed),
                        b=self._rf(1, mul(depth, hd)),
                        acc=self._rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
            self.st(self._rf(2, hd), hbm(add(context.offset, mul(head, hd)), hd))

    def finish(self):
        self.emit("WG.END", wg=self.wg)

    def store_slice(self, source: Tensor, dest: int, rows: int, width: int, col_lo: int, col_hi: int, name: str):
        count = col_hi - col_lo
        with self.loop(name, 0, rows) as row:
            self.ld(hbm(add(source.offset, mul(row, width), col_lo), count), self._rf(0, count))
            self.st(self._rf(0, count), hbm(add(dest, mul(row, width), col_lo), count))



if __name__ == "__main__":
    from pathlib import Path

    from .schedule import generate_m1_d1, generate_m1_p1

    output = Path("programs")
    output.mkdir(exist_ok=True)
    for filename, generate in (("M1_P1.asm", generate_m1_p1), ("M2_D1.asm", generate_m1_d1)):
        program, _, _ = generate()
        (output / filename).write_text(program, encoding="utf-8")
        print(f"Generated {output / filename}: {len(program.splitlines())} lines")
