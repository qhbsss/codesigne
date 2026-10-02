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


def sh(offset, count, shape=None, strides=None, wg="g"):
    result = {"space": "SH", "offset": offset, "count": count, "wg": wg, "lane": 0}
    if shape is not None:
        result.update(shape=shape, strides=strides)
    return result


def imm(value):
    return {"imm": value}


@dataclass(frozen=True)
class Tensor:
    offset: int
    shape: tuple[int, ...]
    space: str = "HBM"
    row_stride: int | None = None
    col_origin: int = 0
    lane: int | None = None
    rf_chunks: tuple[tuple[int, int, int, int], ...] | None = None


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
    def __init__(self, layout: Layout, wg: str = "g", sm: int = 0, memory: Scratch | None = None, lines: list[str] | None = None, gemm_n_tile: int = 64, shared_bytes: int = 0):
        if gemm_n_tile not in (8, 16, 32, 64):
            raise ValueError("Unsupported GEMM N tile")
        self.layout = layout
        self.wg = wg
        self.gemm_n_tile = gemm_n_tile
        self.lines = [] if lines is None else lines
        self.loops = []
        self.serial = 0
        self.memory = Scratch(layout.symbols["scratch"].address // 4) if memory is None else memory
        self.tensors = {}
        self.shared_bytes = shared_bytes
        self.shared_cursor = 0
        # Lane 7 is reserved for transient operands.  Persistent D1 weights
        # are first-fit packed into lanes 0..6.
        self.persistent_rf_used = [0] * 7
        self.emit("WG.BEGIN", wg=wg, sm=sm, shared_bytes=shared_bytes)

    def symbol(self, name):
        item = self.layout.symbols[name]
        return Tensor(item.address // 4, item.shape)

    def alloc(self, name, *shape):
        result = self.memory.alloc(*shape)
        self.tensors[name] = result
        return result

    def _rf(self, lane, count, offset=0):
        return rf(lane, count, offset, wg=self.wg)

    def _sh(self, offset, count, shape=None, strides=None):
        return sh(offset, count, shape, strides, wg=self.wg)

    def cache_weight(self, source: Tensor, n_lo: int, n_hi: int) -> Tensor:
        """Copy a compact column slice to persistent workgroup-private SH."""
        if source.space != "HBM" or len(source.shape) != 2:
            raise ValueError("Only HBM matrices can be cached")
        rows, width_total = source.shape
        if not 0 <= n_lo < n_hi <= width_total:
            raise ValueError("Invalid cached weight slice")
        width = n_hi - n_lo
        count = rows * width
        offset = self.shared_cursor
        self.shared_cursor += count
        if self.shared_cursor * 4 > self.shared_bytes:
            raise ValueError("Cached weights exceed workgroup SH quota")
        self.ld(
            hbm(
                source.offset + n_lo,
                count,
                [rows, width],
                [width_total, 1],
            ),
            self._sh(offset, count),
        )
        return Tensor(
            offset,
            source.shape,
            space="SH",
            row_stride=width,
            col_origin=n_lo,
        )

    def cache_weight_rf(self, source: Tensor, n_lo: int, n_hi: int) -> Tensor:
        """Preload one compact D1 weight slice into persistent RF storage."""
        if source.space != "HBM" or len(source.shape) != 2:
            raise ValueError("Only HBM matrices can be cached")
        rows, width_total = source.shape
        if not 0 <= n_lo < n_hi <= width_total:
            raise ValueError("Invalid cached weight slice")
        width = n_hi - n_lo
        k_chunk = 2048 // width
        chunks = []
        for k_start in range(0, rows, k_chunk):
            depth = min(k_chunk, rows - k_start)
            count = depth * width
            lane = next(
                (
                    index
                    for index, used in enumerate(self.persistent_rf_used)
                    if used + count <= 2048
                ),
                None,
            )
            if lane is None:
                raise ValueError("Persistent RF weight cache exceeds seven lanes")
            offset = self.persistent_rf_used[lane]
            self.persistent_rf_used[lane] += count
            self.ld(
                hbm(
                    source.offset + k_start * width_total + n_lo,
                    count,
                    [depth, width],
                    [width_total, 1],
                ),
                self._rf(lane, count, offset),
            )
            chunks.append((lane, offset, k_start, depth))
        return Tensor(
            chunks[0][1],
            source.shape,
            space="RF",
            row_stride=width,
            col_origin=n_lo,
            lane=chunks[0][0] if len(chunks) == 1 else None,
            rf_chunks=tuple(chunks),
        )

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
        tile_n = min(self.gemm_n_tile, n_hi - n_lo)
        if n_lo < 0 or n_hi > n_extent or (n_hi - n_lo) % tile_n:
            raise ValueError("GEMM N tile range")
        if epilogue not in (None, "residual", "bias_gelu", "residual_bias"):
            raise ValueError("Unsupported GEMM epilogue")
        if epilogue in ("bias_gelu", "residual_bias") and bias is None:
            raise ValueError("Fused bias is missing")
        if epilogue in ("residual", "residual_bias") and residual is None:
            raise ValueError("Fused residual is missing")
        if b.space == "RF":
            self._gemm_persistent(
                a,
                b,
                out,
                m_extent,
                k_extent,
                n_extent,
                n_lo,
                n_hi,
                epilogue,
                bias,
                residual,
            )
            return
        # The selected vector_lanes=8 hardware gives each RF lane 2048 words.
        # N=64 uses two 32-row accumulator panels; N<=32 keeps all 64 prompt
        # rows live while a weight tile is reused.
        rf_lane_words = 2048
        panel_rows = 32 if tile_n == 64 else m_extent
        panels = [
            (start, min(m_extent, start + panel_rows), 2 if start == 0 else 7)
            for start in range(0, m_extent, panel_rows)
        ]
        if len(panels) > 2 or panel_rows * tile_n > rf_lane_words:
            raise ValueError("GEMM accumulators exceed available register lanes")
        k_tile = min(k_extent, rf_lane_words // tile_n)
        if k_tile * tile_n > rf_lane_words:
            raise ValueError("GEMM weight tile exceeds a register lane")
        with self.loop(f"{name}n", n_lo, n_hi, tile_n) as col:
            for panel_start, panel_stop, acc_lane in panels:
                with self.loop(f"{name}z{panel_start}", panel_start, panel_stop, tile_m) as row:
                    rows = minimum(tile_m, sub(panel_stop, row))
                    cells = mul(rows, tile_n)
                    self.vec(
                        "add",
                        [imm(0), imm(0)],
                        self._rf(acc_lane, cells, mul(sub(row, panel_start), tile_n)),
                    )

            def load_b(lane, start, depth):
                if b.space == "HBM":
                    source = hbm(
                        add(b.offset, mul(start, n_extent), col),
                        mul(depth, tile_n),
                        [depth, tile_n],
                        [n_extent, 1],
                    )
                elif b.space == "SH":
                    source = self._sh(
                        add(b.offset, mul(start, b.row_stride), sub(col, b.col_origin)),
                        mul(depth, tile_n),
                        [depth, tile_n],
                        [b.row_stride, 1],
                    )
                else:
                    raise ValueError("Unsupported GEMM weight space")
                self.ld(
                    source,
                    self._rf(lane, mul(depth, tile_n)),
                )

            def use_b(lane, start, depth):
                for panel_start, panel_stop, acc_lane in panels:
                    with self.loop(f"{name}m{panel_start}", panel_start, panel_stop, tile_m) as row:
                        rows = minimum(tile_m, sub(panel_stop, row))
                        if a.space == "HBM":
                            source = hbm(
                                add(a.offset, mul(row, k_extent), start),
                                mul(rows, depth),
                                [rows, depth],
                                [k_extent, 1],
                            )
                        elif a.space == "SH":
                            source = self._sh(
                                add(a.offset, mul(row, a.row_stride), start),
                                mul(rows, depth),
                                [rows, depth],
                                [a.row_stride, 1],
                            )
                        else:
                            raise ValueError("Unsupported GEMM input space")
                        self.ld(source, self._rf(0, mul(rows, depth)))
                        self.emit(
                            "MMA.ACC",
                            a=self._rf(0, mul(rows, depth)),
                            b=self._rf(lane, mul(depth, tile_n)),
                            acc=self._rf(
                                acc_lane,
                                mul(rows, tile_n),
                                mul(sub(row, panel_start), tile_n),
                            ),
                            m=rows,
                            n=tile_n,
                            k=depth,
                            event=None,
                        )

            self._pipeline_chunks(k_extent, k_tile, f"{name}k", load_b, use_b)
            if epilogue is None:
                for panel_start, panel_stop, acc_lane in panels:
                    with self.loop(f"{name}s{panel_start}", panel_start, panel_stop, tile_m) as row:
                        rows = minimum(tile_m, sub(panel_stop, row))
                        cells = mul(rows, tile_n)
                        target = (
                            hbm(
                                add(out.offset, mul(row, n_extent), col),
                                cells,
                                [rows, tile_n],
                                [n_extent, 1],
                            )
                            if out.space == "HBM"
                            else self._sh(
                                add(out.offset, mul(row, out.row_stride), sub(col, out.col_origin)),
                                cells,
                                [rows, tile_n],
                                [out.row_stride, 1],
                            )
                        )
                        self.st(
                            self._rf(
                                acc_lane,
                                cells,
                                mul(sub(row, panel_start), tile_n),
                            ),
                            target,
                        )
            else:
                for panel_start, panel_stop, acc_lane in panels:
                    with self.loop(f"{name}s{panel_start}", panel_start, panel_stop, tile_m) as row:
                        rows = minimum(tile_m, sub(panel_stop, row))
                        with self.loop(f"{name}e{panel_start}", 0, rows) as inner:
                            row_index = add(row, inner)
                            self._epilogue_row(
                                out,
                                n_extent,
                                col,
                                row_index,
                                mul(sub(row_index, panel_start), tile_n),
                                tile_n,
                                epilogue,
                                bias,
                                residual,
                                acc_lane=acc_lane,
                            )

    def _gemm_persistent(
        self,
        a: Tensor,
        b: Tensor,
        out: Tensor,
        m_extent: int,
        k_extent: int,
        n_extent: int,
        n_lo: int,
        n_hi: int,
        epilogue: str | None,
        bias: Tensor | None,
        residual: Tensor | None,
    ):
        """Small-M decode GEMM using a full weight slice already resident in RF."""
        width = n_hi - n_lo
        if (
            not 1 <= m_extent <= 8
            or not b.rf_chunks
            or b.row_stride != width
            or b.col_origin != n_lo
        ):
            raise ValueError("Invalid persistent decode GEMM")
        acc_off = 1024
        cells = m_extent * width
        self.vec("add", [imm(0), imm(0)], self._rf(7, cells, acc_off))
        for lane, offset, k_start, depth in b.rf_chunks:
            self.ld(
                hbm(
                    a.offset + k_start,
                    m_extent * depth,
                    [m_extent, depth],
                    [k_extent, 1],
                ),
                self._rf(7, m_extent * depth),
            )
            self.emit(
                "MMA.ACC",
                a=self._rf(7, m_extent * depth),
                b=self._rf(lane, depth * width, offset),
                acc=self._rf(7, cells, acc_off),
                m=m_extent,
                n=width,
                k=depth,
                event=None,
            )
        with self.loop("pmrow", 0, m_extent) as row:
            acc = self._rf(7, width, add(acc_off, mul(row, width)))
            if epilogue is None:
                stored = acc
            elif epilogue == "residual":
                self.ld(
                    hbm(add(residual.offset, mul(row, n_extent), n_lo), width),
                    self._rf(7, width, 0),
                )
                self.vec("add", [acc, self._rf(7, width, 0)], self._rf(7, width, 32))
                stored = self._rf(7, width, 32)
            elif epilogue == "residual_bias":
                self.ld(
                    hbm(add(residual.offset, mul(row, n_extent), n_lo), width),
                    self._rf(7, width, 0),
                )
                self.vec("add", [acc, self._rf(7, width, 0)], self._rf(7, width, 32))
                self.ld(hbm(bias.offset + n_lo, width), self._rf(7, width, 64))
                self.vec(
                    "add",
                    [self._rf(7, width, 32), self._rf(7, width, 64)],
                    self._rf(7, width, 96),
                )
                stored = self._rf(7, width, 96)
            elif epilogue == "bias_gelu":
                self.ld(hbm(bias.offset + n_lo, width), self._rf(7, width, 0))
                self.vec("add", [acc, self._rf(7, width, 0)], self._rf(7, width, 32))
                self._gelu_persistent(width, 32)
                stored = self._rf(7, width, 96)
            else:
                raise ValueError("Unsupported persistent epilogue")
            self.st(stored, hbm(add(out.offset, mul(row, n_extent), n_lo), width))

    def _gelu_persistent(self, n: int, x_off: int):
        """GELU using only transient regions in RF lane 7."""
        x = self._rf(7, n, x_off)
        temp = self._rf(7, n, 64)
        result = self._rf(7, n, 96)
        self.vec("mul", [x, x], temp)
        self.vec("mul", [temp, x], temp)
        self.vec("fma", [temp, imm(0.044715), x], temp)
        self.vec("mul", [temp, imm(sqrt(2 / 3.141592653589793))], temp)
        self.sfu("tanh", temp, temp)
        self.vec("add", [temp, imm(1)], temp)
        self.vec("mul", [x, imm(0.5)], result)
        self.vec("mul", [result, temp], result)

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
        acc_lane: int = 2,
    ):
        """Fold bias, GELU, and residual into one register tile before its only store."""
        acc = self._rf(acc_lane, n, acc_off)
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
        target = (
            hbm(add(out.offset, mul(row_index, n_extent), col), n)
            if out.space == "HBM"
            else self._sh(
                add(out.offset, mul(row_index, out.row_stride), sub(col, out.col_origin)),
                n,
            )
        )
        self.st(stored, target)

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

    def export_decode_component(
        self,
        qkv: Tensor,
        target: Tensor,
        step: int,
        d: int,
        head: int,
        hd: int,
        component: int,
        component_offset: int = 0,
        width: int | None = None,
    ):
        if component not in (1, 2):
            raise ValueError("Only K/V components are exported")
        if width is None:
            width = hd
        source = qkv.offset + component * d + head * hd + component_offset
        destination = (
            target.offset + head * target.shape[2] * hd + step * hd + component_offset
        )
        self.ld(hbm(source, width), self._rf(7, width, 0))
        self.st(self._rf(7, width, 0), hbm(destination, width))

    def export_decode_component_rows(
        self,
        qkv: Tensor,
        target: Tensor,
        rows: int,
        d: int,
        head: int,
        hd: int,
        component: int,
        component_offset: int = 0,
        width: int | None = None,
    ):
        """Export a contiguous batch of generated K/V rows to head-major ABI storage."""
        if component not in (1, 2):
            raise ValueError("Only K/V components are exported")
        if width is None:
            width = hd
        with self.loop("exportrow", 0, rows) as row:
            source = add(
                qkv.offset,
                mul(row, 3 * d),
                component * d,
                head * hd,
                component_offset,
            )
            destination = add(
                target.offset,
                head * target.shape[2] * hd,
                mul(row, hd),
                component_offset,
            )
            self.ld(hbm(source, width), self._rf(7, width, 0))
            self.st(self._rf(7, width, 0), hbm(destination, width))

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

    def online_attention_local(
        self,
        qkv: Tensor,
        history_k: Tensor,
        history_v: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        head: int,
        key_lo: int,
        key_hi: int,
        generated: int,
        include_generated: bool,
        hd: int,
        max_dest: int,
        sum_dest: int,
        context_dest: int,
        name: str,
    ):
        """Produce stable local softmax statistics without spilling scores."""
        span = key_hi - key_lo
        local_total = span + (generated if include_generated else 0)
        self.ld(hbm(qkv.offset + head * hd, hd), self._rf(0, hd))
        self.vec("add", [imm(0), imm(0)], self._rf(4, local_total))

        def score_range(source: Tensor, stride: int, begin: int, count: int, placed_base: int, tag: str):
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
                placed = add(placed_base, sub(key, begin))
                self.emit(
                    "MMA.ACC",
                    a=self._rf(0, hd),
                    b=self._rf(lane, mul(hd, n)),
                    acc=self._rf(4, n, placed),
                    m=1,
                    n=n,
                    k=hd,
                    event=None,
                )
                self.vec(
                    "mul",
                    [self._rf(4, n, placed), imm(1 / sqrt(hd))],
                    self._rf(4, n, placed),
                )

            self._pipeline_chunks(count, 16, name + tag, load_k, use_k, origin=begin)

        score_range(history_k, history_k.shape[-2], key_lo, span, 0, "k")
        if include_generated:
            score_range(new_k, new_k.shape[2], 0, generated, span, "nk")

        self.reduce("max", self._rf(4, local_total), self._rf(3, 1))
        self.vec("sub", [self._rf(4, local_total), self._rf(3, 1)], self._rf(4, local_total))
        self.sfu("exp", self._rf(4, local_total), self._rf(4, local_total))
        self.reduce("sum", self._rf(4, local_total), self._rf(5, 1))
        self.vec("add", [imm(0), imm(0)], self._rf(2, hd))

        def context_range(source: Tensor, stride: int, begin: int, count: int, placed_base: int, tag: str):
            def load_v(lane, key, n):
                self.ld(
                    hbm(add(source.offset, head * stride * hd, mul(key, hd)), mul(n, hd)),
                    self._rf(lane, mul(n, hd)),
                )

            def use_v(lane, key, n):
                placed = add(placed_base, sub(key, begin))
                self.emit(
                    "MMA.ACC",
                    a=self._rf(4, n, placed),
                    b=self._rf(lane, mul(n, hd)),
                    acc=self._rf(2, hd),
                    m=1,
                    n=hd,
                    k=n,
                    event=None,
                )

            self._pipeline_chunks(count, 16, name + tag, load_v, use_v, origin=begin)

        context_range(history_v, history_v.shape[-2], key_lo, span, 0, "v")
        if include_generated:
            context_range(new_v, new_v.shape[2], 0, generated, span, "nv")
        self.st(self._rf(3, 1), hbm(max_dest, 1))
        self.st(self._rf(5, 1), hbm(sum_dest, 1))
        self.st(self._rf(2, hd), hbm(context_dest, hd))

    def combine_online_attention(
        self,
        max_base: int,
        sum_base: int,
        context_base: int,
        dest: int,
        parts: int,
        hd: int,
    ):
        """Merge local (max, exp-sum, weighted-context) tuples stably."""
        self.ld(hbm(max_base, parts), self._rf(0, parts))
        self.reduce("max", self._rf(0, parts), self._rf(1, 1))
        self.vec("add", [imm(0), imm(0)], self._rf(4, 1))
        self.vec("add", [imm(0), imm(0)], self._rf(2, hd))
        for part in range(parts):
            self.ld(hbm(max_base + part, 1), self._rf(3, 1))
            self.vec("sub", [self._rf(3, 1), self._rf(1, 1)], self._rf(3, 1))
            self.sfu("exp", self._rf(3, 1), self._rf(3, 1))
            self.ld(hbm(sum_base + part, 1), self._rf(5, 1))
            self.vec("mul", [self._rf(5, 1), self._rf(3, 1)], self._rf(5, 1))
            self.vec("add", [self._rf(4, 1), self._rf(5, 1)], self._rf(4, 1))
            self.ld(hbm(context_base + part * hd, hd), self._rf(0, hd))
            self.vec("mul", [self._rf(0, hd), self._rf(3, 1)], self._rf(0, hd))
            self.vec("add", [self._rf(2, hd), self._rf(0, hd)], self._rf(2, hd))
        self.vec("div", [self._rf(2, hd), self._rf(4, 1)], self._rf(2, hd))
        self.st(self._rf(2, hd), hbm(dest, hd))

    def online_attention_local_persistent(
        self,
        qkv: Tensor,
        history_k: Tensor,
        history_v: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        head: int,
        key_lo: int,
        key_hi: int,
        generated: int,
        include_generated: bool,
        hd: int,
        max_dest: int,
        sum_dest: int,
        context_dest: int,
    ):
        """RF-cache-safe local attention using lane 7 and lane-1 high offsets."""
        span = key_hi - key_lo
        local_total = span + (generated if include_generated else 0)
        # A 48-key staging tile occupies lane-7 words 64..1599.  Keep the
        # complete local score vector and reduction state above it so a
        # multi-tile 96-key partition cannot overwrite earlier scores.
        score_off, max_off, sum_off, ctx_off = 1600, 1720, 1721, 1760
        self.ld(hbm(qkv.offset + head * hd, hd), self._rf(7, hd, 0))
        self.vec("add", [imm(0), imm(0)], self._rf(7, local_total, score_off))

        def score(source: Tensor, stride: int, begin: int, count: int, placed: int):
            for chunk_start in range(0, count, 48):
                chunk = min(48, count - chunk_start)
                chunk_placed = placed + chunk_start
                self.ld(
                    hbm(
                        source.offset + head * stride * hd + (begin + chunk_start) * hd,
                        hd * chunk,
                        [hd, chunk],
                        [1, hd],
                    ),
                    self._rf(7, hd * chunk, 64),
                )
                self.emit(
                    "MMA.ACC",
                    a=self._rf(7, hd, 0),
                    b=self._rf(7, hd * chunk, 64),
                    acc=self._rf(7, chunk, score_off + chunk_placed),
                    m=1,
                    n=chunk,
                    k=hd,
                    event=None,
                )
                self.vec(
                    "mul",
                    [self._rf(7, chunk, score_off + chunk_placed), imm(1 / sqrt(hd))],
                    self._rf(7, chunk, score_off + chunk_placed),
                )

        score(history_k, history_k.shape[-2], key_lo, span, 0)
        if include_generated:
            score(new_k, new_k.shape[2], 0, generated, span)
        self.reduce("max", self._rf(7, local_total, score_off), self._rf(7, 1, max_off))
        self.vec(
            "sub",
            [self._rf(7, local_total, score_off), self._rf(7, 1, max_off)],
            self._rf(7, local_total, score_off),
        )
        self.sfu(
            "exp",
            self._rf(7, local_total, score_off),
            self._rf(7, local_total, score_off),
        )
        self.reduce("sum", self._rf(7, local_total, score_off), self._rf(7, 1, sum_off))
        self.vec("add", [imm(0), imm(0)], self._rf(7, hd, ctx_off))

        def context(source: Tensor, stride: int, begin: int, count: int, placed: int):
            for chunk_start in range(0, count, 48):
                chunk = min(48, count - chunk_start)
                chunk_placed = placed + chunk_start
                self.ld(
                    hbm(
                        source.offset + head * stride * hd + (begin + chunk_start) * hd,
                        chunk * hd,
                    ),
                    self._rf(7, chunk * hd, 64),
                )
                self.emit(
                    "MMA.ACC",
                    a=self._rf(7, chunk, score_off + chunk_placed),
                    b=self._rf(7, chunk * hd, 64),
                    acc=self._rf(7, hd, ctx_off),
                    m=1,
                    n=hd,
                    k=chunk,
                    event=None,
                )

        context(history_v, history_v.shape[-2], key_lo, span, 0)
        if include_generated:
            context(new_v, new_v.shape[2], 0, generated, span)
        self.st(self._rf(7, 1, max_off), hbm(max_dest, 1))
        self.st(self._rf(7, 1, sum_off), hbm(sum_dest, 1))
        self.st(self._rf(7, hd, ctx_off), hbm(context_dest, hd))

    def combine_online_attention_persistent(
        self,
        max_base: int,
        sum_base: int,
        context_base: int,
        dest: int,
        parts: int,
        hd: int,
    ):
        max_off, run_sum_off, ctx_off = 650, 651, 700
        self.ld(hbm(max_base, parts), self._rf(7, parts, 0))
        self.reduce("max", self._rf(7, parts, 0), self._rf(7, 1, max_off))
        self.vec("add", [imm(0), imm(0)], self._rf(7, 1, run_sum_off))
        self.vec("add", [imm(0), imm(0)], self._rf(7, hd, ctx_off))
        for part in range(parts):
            self.ld(hbm(max_base + part, 1), self._rf(7, 1, 0))
            self.vec(
                "sub",
                [self._rf(7, 1, 0), self._rf(7, 1, max_off)],
                self._rf(7, 1, 0),
            )
            self.sfu("exp", self._rf(7, 1, 0), self._rf(7, 1, 0))
            self.ld(hbm(sum_base + part, 1), self._rf(7, 1, 1))
            self.vec("mul", [self._rf(7, 1, 1), self._rf(7, 1, 0)], self._rf(7, 1, 1))
            self.vec(
                "add",
                [self._rf(7, 1, run_sum_off), self._rf(7, 1, 1)],
                self._rf(7, 1, run_sum_off),
            )
            self.ld(hbm(context_base + part * hd, hd), self._rf(7, hd, 32))
            self.vec(
                "mul",
                [self._rf(7, hd, 32), self._rf(7, 1, 0)],
                self._rf(7, hd, 32),
            )
            self.vec(
                "add",
                [self._rf(7, hd, ctx_off), self._rf(7, hd, 32)],
                self._rf(7, hd, ctx_off),
            )
        self.vec(
            "div",
            [self._rf(7, hd, ctx_off), self._rf(7, 1, run_sum_off)],
            self._rf(7, hd, ctx_off),
        )
        self.st(self._rf(7, hd, ctx_off), hbm(dest, hd))

    def layernorm(
        self,
        source: Tensor,
        gamma: Tensor,
        beta: Tensor,
        out: Tensor,
        rows: int,
        d: int,
        name: str,
        row_lo: int = 0,
        row_hi: int | None = None,
    ):
        if row_hi is None:
            row_hi = rows
        if not 0 <= row_lo <= row_hi <= rows:
            raise ValueError("Invalid LayerNorm row slice")
        self.ld(hbm(gamma.offset, d), self._rf(3, d))
        self.ld(hbm(beta.offset, d), self._rf(4, d))
        with self.loop(name, row_lo, row_hi) as row:
            base = add(source.offset, mul(row, d))
            target = (
                hbm(add(out.offset, mul(row, d)), d)
                if out.space == "HBM"
                else self._sh(add(out.offset, mul(row, out.row_stride)), d)
            )
            self.ld(hbm(base, d), self._rf(0, d))
            self.reduce("sum", self._rf(0, d), self._rf(1, 1))
            self.vec("mul", [self._rf(1, 1), imm(1 / d)], self._rf(1, 1))
            self.vec("sub", [self._rf(0, d), self._rf(1, 1)], self._rf(2, d))
            self.vec("mul", [self._rf(2, d), self._rf(2, d)], self._rf(5, d))
            self.reduce("sum", self._rf(5, d), self._rf(1, 1))
            self.vec("mul", [self._rf(1, 1), imm(1 / d)], self._rf(1, 1))
            self.vec("add", [self._rf(1, 1), imm(1e-5)], self._rf(1, 1))
            self.sfu("rsqrt", self._rf(1, 1), self._rf(1, 1))
            self.vec("mul", [self._rf(2, d), self._rf(1, 1)], self._rf(2, d))
            self.vec("mul", [self._rf(2, d), self._rf(3, d)], self._rf(2, d))
            self.vec("add", [self._rf(2, d), self._rf(4, d)], self._rf(2, d))
            self.st(self._rf(2, d), target)

    def layernorm_persistent(
        self, source: Tensor, gamma: Tensor, beta: Tensor, out: Tensor, d: int
    ):
        """Decode LayerNorm confined to transient RF regions."""
        scalar = 700
        self.ld(hbm(source.offset, d), self._rf(7, d, 0))
        self.reduce("sum", self._rf(7, d, 0), self._rf(7, 1, scalar))
        self.vec(
            "mul",
            [self._rf(7, 1, scalar), imm(1 / d)],
            self._rf(7, 1, scalar),
        )
        self.vec(
            "sub",
            [self._rf(7, d, 0), self._rf(7, 1, scalar)],
            self._rf(7, d, 128),
        )
        self.vec(
            "mul",
            [self._rf(7, d, 128), self._rf(7, d, 128)],
            self._rf(7, d, 256),
        )
        self.reduce("sum", self._rf(7, d, 256), self._rf(7, 1, scalar))
        self.vec(
            "mul",
            [self._rf(7, 1, scalar), imm(1 / d)],
            self._rf(7, 1, scalar),
        )
        self.vec(
            "add", [self._rf(7, 1, scalar), imm(1e-5)], self._rf(7, 1, scalar)
        )
        self.sfu("rsqrt", self._rf(7, 1, scalar), self._rf(7, 1, scalar))
        self.vec(
            "mul",
            [self._rf(7, d, 128), self._rf(7, 1, scalar)],
            self._rf(7, d, 128),
        )
        self.ld(hbm(gamma.offset, d), self._rf(7, d, 384))
        self.vec(
            "mul",
            [self._rf(7, d, 128), self._rf(7, d, 384)],
            self._rf(7, d, 128),
        )
        self.ld(hbm(beta.offset, d), self._rf(7, d, 512))
        self.vec(
            "add",
            [self._rf(7, d, 128), self._rf(7, d, 512)],
            self._rf(7, d, 128),
        )
        self.st(self._rf(7, d, 128), hbm(out.offset, d))

    def layernorm_persistent_rows(
        self,
        source: Tensor,
        gamma: Tensor,
        beta: Tensor,
        out: Tensor,
        rows: int,
        d: int,
        name: str,
    ):
        """Apply decode LayerNorm to several contiguous rows with one workgroup."""
        scalar = 700
        with self.loop(name, 0, rows) as row:
            source_offset = add(source.offset, mul(row, d))
            out_offset = add(out.offset, mul(row, d))
            self.ld(hbm(source_offset, d), self._rf(7, d, 0))
            self.reduce("sum", self._rf(7, d, 0), self._rf(7, 1, scalar))
            self.vec(
                "mul", [self._rf(7, 1, scalar), imm(1 / d)], self._rf(7, 1, scalar)
            )
            self.vec(
                "sub",
                [self._rf(7, d, 0), self._rf(7, 1, scalar)],
                self._rf(7, d, 128),
            )
            self.vec(
                "mul",
                [self._rf(7, d, 128), self._rf(7, d, 128)],
                self._rf(7, d, 256),
            )
            self.reduce("sum", self._rf(7, d, 256), self._rf(7, 1, scalar))
            self.vec(
                "mul", [self._rf(7, 1, scalar), imm(1 / d)], self._rf(7, 1, scalar)
            )
            self.vec(
                "add", [self._rf(7, 1, scalar), imm(1e-5)], self._rf(7, 1, scalar)
            )
            self.sfu("rsqrt", self._rf(7, 1, scalar), self._rf(7, 1, scalar))
            self.vec(
                "mul",
                [self._rf(7, d, 128), self._rf(7, 1, scalar)],
                self._rf(7, d, 128),
            )
            self.ld(hbm(gamma.offset, d), self._rf(7, d, 384))
            self.vec(
                "mul",
                [self._rf(7, d, 128), self._rf(7, d, 384)],
                self._rf(7, d, 128),
            )
            self.ld(hbm(beta.offset, d), self._rf(7, d, 512))
            self.vec(
                "add",
                [self._rf(7, d, 128), self._rf(7, d, 512)],
                self._rf(7, d, 128),
            )
            self.st(self._rf(7, d, 128), hbm(out_offset, d))

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
        """Compute causal attention with true 16x16 score/context tiles.

        Scores remain row-contiguous in RF lane 4 for the softmax.  Lane 3 is
        a compact tile staging area: it first receives the 16x16 score MMA,
        then gathers the corresponding probability tile for the context MMA.
        This turns sixteen m=1 MMAs and sixteen diagonal K/V reloads into one
        m=16 MMA per tile without changing the K reduction order.
        """
        stride = rows
        scale = 1 / sqrt(hd)
        seq = k_out.shape[2]
        for head in range(head_lo, head_hi):
            k_base = k_out.offset + head * seq * hd
            v_base = v_out.offset + head * v_out.shape[2] * hd
            with self.loop(f"{name}b{head}", 0, rows // 16) as block_i:
                block = mul(block_i, 16)
                self.ld(
                    hbm(
                        add(qkv.offset, mul(block, 3 * d), head * hd),
                        16 * hd,
                        [16, hd],
                        [3 * d, 1],
                    ),
                    self._rf(0, 16 * hd),
                )

                # Completed tiles to the left of the causal diagonal.
                with self.loop(f"{name}t{head}", 0, block_i) as tile:
                    key = mul(tile, 16)
                    self.ld(
                        hbm(add(k_base, mul(key, hd)), hd * 16, [hd, 16], [1, hd]),
                        self._rf(1, hd * 16),
                    )
                    self.vec("add", [imm(0), imm(0)], self._rf(3, 16 * 16))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(0, 16 * hd),
                        b=self._rf(1, hd * 16),
                        acc=self._rf(3, 16 * 16),
                        m=16,
                        n=16,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [self._rf(3, 16 * 16), imm(scale)], self._rf(3, 16 * 16))
                    with self.loop(f"{name}r{head}", 0, 16) as local:
                        slot = add(mul(local, stride), key)
                        self.vec(
                            "add",
                            [self._rf(3, 16, mul(local, 16)), imm(0)],
                            self._rf(4, 16, slot),
                        )

                # Full diagonal tile followed by an explicit strict-upper mask.
                self.ld(
                    hbm(add(k_base, mul(block, hd)), hd * 16, [hd, 16], [1, hd]),
                    self._rf(1, hd * 16),
                )
                self.vec("add", [imm(0), imm(0)], self._rf(3, 16 * 16))
                self.emit(
                    "MMA.ACC",
                    a=self._rf(0, 16 * hd),
                    b=self._rf(1, hd * 16),
                    acc=self._rf(3, 16 * 16),
                    m=16,
                    n=16,
                    k=hd,
                    event=None,
                )
                self.vec("mul", [self._rf(3, 16 * 16), imm(scale)], self._rf(3, 16 * 16))
                with self.loop(f"{name}p{head}", 0, 16) as local:
                    slot = add(mul(local, stride), block)
                    self.vec(
                        "add",
                        [self._rf(3, 16, mul(local, 16)), imm(0)],
                        self._rf(4, 16, slot),
                    )
                with self.loop(f"{name}mask{head}", 0, 15) as local:
                    self.vec(
                        "add",
                        [imm(-1e30), imm(0)],
                        self._rf(
                            4,
                            sub(15, local),
                            add(mul(local, stride), block, local, 1),
                        ),
                    )

                with self.loop(f"{name}s{head}", 0, 16) as local:
                    limit = add(block, local, 1)
                    slot = mul(local, stride)
                    self.reduce("max", self._rf(4, limit, slot), self._rf(1, 1))
                    self.vec("sub", [self._rf(4, limit, slot), self._rf(1, 1)], self._rf(4, limit, slot))
                    self.sfu("exp", self._rf(4, limit, slot), self._rf(4, limit, slot))
                    self.reduce("sum", self._rf(4, limit, slot), self._rf(1, 1))
                    self.vec("div", [self._rf(4, limit, slot), self._rf(1, 1)], self._rf(4, limit, slot))

                # The row-wise softmax above intentionally stops at the causal
                # limit.  Clear the untouched strict-upper cells before the
                # dense 16x16 probability/value MMA consumes the whole tile.
                with self.loop(f"{name}zero{head}", 0, 15) as local:
                    self.vec(
                        "add",
                        [imm(0), imm(0)],
                        self._rf(
                            4,
                            sub(15, local),
                            add(mul(local, stride), block, local, 1),
                        ),
                    )

                self.vec("add", [imm(0), imm(0)], self._rf(2, 16 * hd))
                with self.loop(f"{name}u{head}", 0, add(block_i, 1)) as tile:
                    key = mul(tile, 16)
                    with self.loop(f"{name}g{head}", 0, 16) as local:
                        self.vec(
                            "add",
                            [self._rf(4, 16, add(mul(local, stride), key)), imm(0)],
                            self._rf(3, 16, mul(local, 16)),
                        )
                    self.ld(hbm(add(v_base, mul(key, hd)), 16 * hd), self._rf(1, 16 * hd))
                    self.emit(
                        "MMA.ACC",
                        a=self._rf(3, 16 * 16),
                        b=self._rf(1, 16 * hd),
                        acc=self._rf(2, 16 * hd),
                        m=16,
                        n=hd,
                        k=16,
                        event=None,
                    )
                with self.loop(f"{name}y{head}", 0, 16) as local:
                    row = add(block, local)
                    self.st(
                        self._rf(2, hd, mul(local, hd)),
                        hbm(add(context.offset, mul(row, d), head * hd), hd),
                    )

    def export_prompt_kv_sh(
        self,
        qkv: Tensor,
        k_out: Tensor,
        v_out: Tensor,
        row_base: int,
        rows: int,
        d: int,
        h: int,
        hd: int,
    ):
        """Export one row-owned SH QKV tile to the public head-major K/V ABI."""
        if qkv.space != "SH" or qkv.row_stride != 3 * d:
            raise ValueError("Prompt QKV tile must be row-major shared memory")
        seq = k_out.shape[2]
        with self.loop("pkvh", 0, h) as head:
            with self.loop("pkvr", 0, rows) as local:
                global_row = add(row_base, local)
                for target, component in ((k_out, 1), (v_out, 2)):
                    self.ld(
                        self._sh(
                            add(
                                qkv.offset,
                                mul(local, qkv.row_stride),
                                component * d,
                                mul(head, hd),
                            ),
                            hd,
                        ),
                        self._rf(0, hd),
                    )
                    self.st(
                        self._rf(0, hd),
                        hbm(
                            add(target.offset, mul(head, seq * hd), mul(global_row, hd)),
                            hd,
                        ),
                    )

    def attention_prompt_row_block(
        self,
        qkv: Tensor,
        context: Tensor,
        k_out: Tensor,
        v_out: Tensor,
        row_base: int,
        rows: int,
        d: int,
        h: int,
        hd: int,
        name: str,
    ):
        """Causal prompt attention for one eight-row tile and all heads.

        The worker owns rows rather than heads. Scores and contexts use m=8
        MMA tiles, while public K/V remains head-major for the challenge ABI.
        """
        if qkv.space != "SH" or context.space != "SH" or rows != 8 or hd != 32:
            raise ValueError("Unsupported row-owned prompt attention tile")
        limit_max = row_base + rows
        seq = k_out.shape[2]
        score_stride = 64
        with self.loop(name + "h", 0, h) as head:
            self.ld(
                self._sh(
                    add(qkv.offset, mul(head, hd)),
                    rows * hd,
                    [rows, hd],
                    [qkv.row_stride, 1],
                ),
                self._rf(0, rows * hd),
            )
            with self.loop(name + "k", 0, limit_max, 16) as key:
                depth = minimum(16, sub(limit_max, key))
                self.ld(
                    hbm(
                        add(k_out.offset, mul(head, seq * hd), mul(key, hd)),
                        mul(hd, depth),
                        [hd, depth],
                        [1, hd],
                    ),
                    self._rf(1, mul(hd, depth)),
                )
                self.vec("add", [imm(0), imm(0)], self._rf(3, mul(rows, depth)))
                self.emit(
                    "MMA.ACC",
                    a=self._rf(0, rows * hd),
                    b=self._rf(1, mul(hd, depth)),
                    acc=self._rf(3, mul(rows, depth)),
                    m=rows,
                    n=depth,
                    k=hd,
                    event=None,
                )
                self.vec(
                    "mul",
                    [self._rf(3, mul(rows, depth)), imm(1 / sqrt(hd))],
                    self._rf(3, mul(rows, depth)),
                )
                with self.loop(name + "copy", 0, rows) as local:
                    self.vec(
                        "add",
                        [self._rf(3, depth, mul(local, depth)), imm(0)],
                        self._rf(4, depth, add(mul(local, score_stride), key)),
                    )
            with self.loop(name + "mask", 0, rows - 1) as local:
                limit = add(row_base, local, 1)
                masked = sub(limit_max, limit)
                self.vec(
                    "add",
                    [imm(-1e30), imm(0)],
                    self._rf(4, masked, add(mul(local, score_stride), limit)),
                )
            with self.loop(name + "softmax", 0, rows) as local:
                limit = add(row_base, local, 1)
                score = self._rf(4, limit, mul(local, score_stride))
                self.reduce("max", score, self._rf(1, 1))
                self.vec("sub", [score, self._rf(1, 1)], score)
                self.sfu("exp", score, score)
                self.reduce("sum", score, self._rf(1, 1))
                self.vec("div", [score, self._rf(1, 1)], score)
            with self.loop(name + "zero", 0, rows - 1) as local:
                limit = add(row_base, local, 1)
                masked = sub(limit_max, limit)
                self.vec(
                    "add",
                    [imm(0), imm(0)],
                    self._rf(4, masked, add(mul(local, score_stride), limit)),
                )
            self.vec("add", [imm(0), imm(0)], self._rf(2, rows * hd))
            with self.loop(name + "v", 0, limit_max, 16) as key:
                depth = minimum(16, sub(limit_max, key))
                with self.loop(name + "gather", 0, rows) as local:
                    self.vec(
                        "add",
                        [self._rf(4, depth, add(mul(local, score_stride), key)), imm(0)],
                        self._rf(3, depth, mul(local, depth)),
                    )
                self.ld(
                    hbm(
                        add(v_out.offset, mul(head, seq * hd), mul(key, hd)),
                        mul(depth, hd),
                    ),
                    self._rf(1, mul(depth, hd)),
                )
                self.emit(
                    "MMA.ACC",
                    a=self._rf(3, mul(rows, depth)),
                    b=self._rf(1, mul(depth, hd)),
                    acc=self._rf(2, rows * hd),
                    m=rows,
                    n=hd,
                    k=depth,
                    event=None,
                )
            with self.loop(name + "store", 0, rows) as local:
                self.st(
                    self._rf(2, hd, mul(local, hd)),
                    self._sh(
                        add(context.offset, mul(local, context.row_stride), mul(head, hd)),
                        hd,
                    ),
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

    def store_slice_persistent(
        self, source: Tensor, dest: int, width: int, col_lo: int, col_hi: int
    ):
        count = col_hi - col_lo
        self.ld(hbm(source.offset + col_lo, count), self._rf(7, count, 0))
        self.st(self._rf(7, count, 0), hbm(dest + col_lo, count))



if __name__ == "__main__":
    from pathlib import Path

    from .schedule import generate_m1_d1, generate_m1_p1

    output = Path("programs")
    output.mkdir(exist_ok=True)
    for filename, generate in (("M1_P1.asm", generate_m1_p1), ("M2_D1.asm", generate_m1_d1)):
        program, _, _ = generate()
        (output / filename).write_text(program, encoding="utf-8")
        print(f"Generated {output / filename}: {len(program.splitlines())} lines")
