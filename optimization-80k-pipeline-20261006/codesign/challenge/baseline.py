"""Generate literal-ISA baselines for the two active trial cases.

The Python here is a compiler only.  The emitted program uses no operator-level
instructions for LayerNorm, attention, softmax, or GELU.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt

from .abi import Layout, build_layout
from .workload import MODELS, SCENARIOS


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


def rf(lane, count, offset=0):
    return {"space": "RF", "offset": offset, "count": count, "wg": "g", "lane": lane}


def imm(value):
    return {"imm": value}


@dataclass(frozen=True)
class Tensor:
    offset: int
    shape: tuple[int, ...]


class Builder:
    def __init__(self, layout: Layout, gemm_n_tile: int = 16):
        if gemm_n_tile not in (8, 16):
            raise ValueError("Unsupported GEMM N tile")
        self.layout = layout
        self.gemm_n_tile = gemm_n_tile
        self.lines = []
        self.loops = []
        self.serial = 0
        self.cursor = layout.symbols["scratch"].address // 4
        self.tensors = {}
        self.emit("WG.BEGIN", wg="g", sm=0, shared_bytes=0)

    def symbol(self, name):
        item = self.layout.symbols[name]
        return Tensor(item.address // 4, item.shape)

    def alloc(self, name, *shape):
        size = 1
        for extent in shape:
            size *= extent
        result = Tensor(self.cursor, tuple(shape))
        self.cursor += size
        self.tensors[name] = result
        return result

    def emit(self, op, **args):
        if "event" in args and args["event"] is None:
            self.serial += 1
            suffix = "".join(f"_{{{name}}}" for name in self.loops)
            args["event"] = f"e{self.serial}{suffix}"
        self.lines.append(f"{op} {json.dumps(args, separators=(',', ':'))}")

    @contextmanager
    def loop(self, name, start, stop, step=1):
        self.emit("FOR", var=name, start=start, stop=stop, step=step)
        self.loops.append(name)
        yield var(name)
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

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        out: Tensor,
        m_extent: int,
        k_extent: int,
        n_extent: int,
        name: str,
    ):
        if b.shape != (k_extent, n_extent):
            raise ValueError("GEMM weight shape")
        tile_n = self.gemm_n_tile
        if n_extent % tile_n:
            raise ValueError("GEMM N extent does not fit tile")
        with self.loop(f"{name}m", 0, m_extent, 8) as row:
            rows = minimum(8, sub(m_extent, row))
            with self.loop(f"{name}n", 0, n_extent, tile_n) as col:
                cells = mul(rows, tile_n)
                self.vec("add", [imm(0), imm(0)], rf(2, cells))
                with self.loop(f"{name}k", 0, k_extent, 48) as inner:
                    depth = minimum(48, sub(k_extent, inner))
                    self.ld(
                        hbm(
                            add(a.offset, mul(row, k_extent), inner),
                            mul(rows, depth),
                            [rows, depth],
                            [k_extent, 1],
                        ),
                        rf(0, mul(rows, depth)),
                    )
                    self.ld(
                        hbm(
                            add(b.offset, mul(inner, n_extent), col),
                            mul(depth, tile_n),
                            [depth, tile_n],
                            [n_extent, 1],
                        ),
                        rf(1, mul(depth, tile_n)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, mul(rows, depth)),
                        b=rf(1, mul(depth, tile_n)),
                        acc=rf(2, cells),
                        m=rows,
                        n=tile_n,
                        k=depth,
                        event=None,
                    )
                self.st(
                    rf(2, cells),
                    hbm(
                        add(out.offset, mul(row, n_extent), col),
                        cells,
                        [rows, tile_n],
                        [n_extent, 1],
                    ),
                )

    def layernorm(
        self, source: Tensor, gamma: Tensor, beta: Tensor, out: Tensor, rows: int, d: int, name: str
    ):
        with self.loop(name, 0, rows) as row:
            base = add(source.offset, mul(row, d))
            target = add(out.offset, mul(row, d))
            self.ld(hbm(base, d), rf(0, d))
            self.reduce("sum", rf(0, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("sub", [rf(0, d), rf(1, 1)], rf(2, d))
            self.vec("mul", [rf(2, d), rf(2, d)], rf(3, d))
            self.reduce("sum", rf(3, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("add", [rf(1, 1), imm(1e-5)], rf(1, 1))
            self.sfu("rsqrt", rf(1, 1), rf(1, 1))
            self.vec("mul", [rf(2, d), rf(1, 1)], rf(2, d))
            self.ld(hbm(gamma.offset, d), rf(3, d))
            self.vec("mul", [rf(2, d), rf(3, d)], rf(2, d))
            self.ld(hbm(beta.offset, d), rf(3, d))
            self.vec("add", [rf(2, d), rf(3, d)], rf(2, d))
            self.st(rf(2, d), hbm(target, d))

    def add_rows(
        self,
        left: Tensor,
        right: Tensor,
        out: Tensor,
        rows: int,
        width: int,
        name: str,
        bias: Tensor | None = None,
    ):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                self.ld(hbm(add(left.offset, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(right.offset, mul(row, width), col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                if bias is not None:
                    self.ld(hbm(add(bias.offset, col), n), rf(3, n))
                    self.vec("add", [rf(2, n), rf(3, n)], rf(2, n))
                self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def add_bias(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                self.ld(hbm(add(source.offset, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(bias.offset, col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def gelu(self, source: Tensor, out: Tensor, rows: int, width: int, name: str):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                addr = add(source.offset, mul(row, width), col)
                self.ld(hbm(addr, n), rf(0, n))
                self.vec("mul", [rf(0, n), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), rf(0, n)], rf(1, n))
                self.vec("fma", [rf(1, n), imm(0.044715), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), imm(sqrt(2 / 3.141592653589793))], rf(1, n))
                self.sfu("tanh", rf(1, n), rf(1, n))
                self.vec("add", [rf(1, n), imm(1)], rf(1, n))
                self.vec("mul", [rf(0, n), imm(0.5)], rf(0, n))
                self.vec("mul", [rf(0, n), rf(1, n)], rf(0, n))
                self.st(rf(0, n), hbm(add(out.offset, mul(row, width), col), n))

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
    ):
        # Export every layer's K/V in the public head-major ABI.
        with self.loop(f"{name}h", 0, h) as head:
            with self.loop(f"{name}r", 0, rows) as row:
                for kind, target, component in (("k", k_out, 1), ("v", v_out, 2)):
                    source_addr = add(qkv.offset, mul(row, 3 * d), component * d, mul(head, hd))
                    target_addr = add(
                        target.offset, mul(head, target.shape[2] * hd), mul(add(past, row), hd)
                    )
                    self.ld(hbm(source_addr, hd), rf(0, hd))
                    self.st(rf(0, hd), hbm(target_addr, hd))
        total = past + rows
        scores = self.alloc(f"{name}_scores", rows, total)
        probs = self.alloc(f"{name}_probs", rows, total)
        with self.loop(f"{name}q", 0, rows) as row:
            limit = add(past, row, 1)
            with self.loop(f"{name}head", 0, h) as head:
                q_addr = add(qkv.offset, mul(row, 3 * d), mul(head, hd))
                self.ld(hbm(q_addr, hd), rf(0, hd))
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
                        rf(1, cells),
                    )
                    self.vec("add", [imm(0), imm(0)], rf(2, n))
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, hd),
                        b=rf(1, cells),
                        acc=rf(2, n),
                        m=1,
                        n=n,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [rf(2, n), imm(1 / sqrt(hd))], rf(2, n))
                    self.st(rf(2, n), hbm(add(scores.offset, mul(row, total), key), n))
                self.ld(hbm(add(scores.offset, mul(row, total)), limit), rf(0, limit))
                self.reduce("max", rf(0, limit), rf(1, 1))
                self.vec("sub", [rf(0, limit), rf(1, 1)], rf(0, limit))
                self.sfu("exp", rf(0, limit), rf(0, limit))
                self.reduce("sum", rf(0, limit), rf(1, 1))
                self.vec("div", [rf(0, limit), rf(1, 1)], rf(0, limit))
                self.st(rf(0, limit), hbm(add(probs.offset, mul(row, total)), limit))
                self.vec("add", [imm(0), imm(0)], rf(2, hd))
                with self.loop(f"{name}value", 0, limit, 16) as key:
                    depth = minimum(16, sub(limit, key))
                    self.ld(hbm(add(probs.offset, mul(row, total), key), depth), rf(0, depth))
                    self.ld(
                        hbm(
                            add(v_out.offset, mul(head, v_out.shape[2] * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, depth),
                        b=rf(1, mul(depth, hd)),
                        acc=rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
                self.st(rf(2, hd), hbm(add(context.offset, mul(row, d), mul(head, hd)), hd))

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
    ):
        """One decode step: score historical and newly generated KV separately."""
        generated = step + 1
        total = past + generated
        scores = self.alloc(f"{name}_scores", total)
        probs = self.alloc(f"{name}_probs", total)
        for kind, target, component in (("k", new_k, 1), ("v", new_v, 2)):
            with self.loop(f"{name}{kind}h", 0, h) as head:
                source = add(qkv.offset, component * d, mul(head, hd))
                destination = add(target.offset, mul(head, target.shape[2] * hd), step * hd)
                self.ld(hbm(source, hd), rf(0, hd))
                self.st(rf(0, hd), hbm(destination, hd))
        with self.loop(f"{name}head", 0, h) as head:
            self.ld(hbm(add(qkv.offset, mul(head, hd)), hd), rf(0, hd))
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
                        rf(1, cells),
                    )
                    self.vec("add", [imm(0), imm(0)], rf(2, n))
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, hd),
                        b=rf(1, cells),
                        acc=rf(2, n),
                        m=1,
                        n=n,
                        k=hd,
                        event=None,
                    )
                    self.vec("mul", [rf(2, n), imm(1 / sqrt(hd))], rf(2, n))
                    self.st(rf(2, n), hbm(add(scores.offset, score_base, key), n))
            self.ld(hbm(scores.offset, total), rf(0, total))
            self.reduce("max", rf(0, total), rf(1, 1))
            self.vec("sub", [rf(0, total), rf(1, 1)], rf(0, total))
            self.sfu("exp", rf(0, total), rf(0, total))
            self.reduce("sum", rf(0, total), rf(1, 1))
            self.vec("div", [rf(0, total), rf(1, 1)], rf(0, total))
            self.st(rf(0, total), hbm(probs.offset, total))
            self.vec("add", [imm(0), imm(0)], rf(2, hd))
            for source, count, prob_base, stride in (
                (history_v, past, 0, past),
                (new_v, generated, past, new_v.shape[2]),
            ):
                with self.loop(f"{name}value{prob_base}", 0, count, 16) as key:
                    depth = minimum(16, sub(count, key))
                    self.ld(hbm(add(probs.offset, prob_base, key), depth), rf(0, depth))
                    self.ld(
                        hbm(
                            add(source.offset, mul(head, stride * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, depth),
                        b=rf(1, mul(depth, hd)),
                        acc=rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
            self.st(rf(2, hd), hbm(add(context.offset, mul(head, hd)), hd))

    def finish(self):
        self.emit("WG.END", wg="g")
        return "\n".join(self.lines) + "\n"


def generate_m1_p1() -> tuple[str, Layout, int]:
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    layout = build_layout(model, "P1")
    b = Builder(layout)
    prompt = b.symbol("input/prompt")
    step = b.symbol("input/step0")
    output = b.symbol("output/hidden")
    # All prompts finish before STEP.COMMIT releases either batch member's first new token.
    for phase, rows in (("p", prompt_rows), ("s", 1)):
        for member in range(batch):
            initial = (
                Tensor(prompt.offset + member * prompt_rows * d, (rows, d))
                if phase == "p"
                else Tensor(step.offset + member * d, (rows, d))
            )
            x = initial
            for layer in range(model.layers):
                prefix = f"layer{layer}/"
                name = f"{phase}b{member}l{layer}"
                weights = {
                    key: b.symbol(prefix + key)
                    for key in (
                        "ln1_g",
                        "ln1_b",
                        "wqkv",
                        "wo",
                        "ln2_g",
                        "ln2_b",
                        "w1",
                        "b1",
                        "w2",
                        "b2",
                    )
                }
                ln1 = b.alloc(f"{name}_ln1", rows, d)
                qkv = b.alloc(f"{name}_qkv", rows, 3 * d)
                ctx = b.alloc(f"{name}_ctx", rows, d)
                att = b.alloc(f"{name}_att", rows, d)
                res = b.alloc(f"{name}_res", rows, d)
                ln2 = b.alloc(f"{name}_ln2", rows, d)
                ff1 = b.alloc(f"{name}_ff1", rows, f)
                biased = b.alloc(f"{name}_biased", rows, f)
                activated = b.alloc(f"{name}_gelu", rows, f)
                ff2 = b.alloc(f"{name}_ff2", rows, d)
                y = b.alloc(f"{name}_out", rows, d)
                b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, rows, d, name + "ln1")
                b.gemm(ln1, weights["wqkv"], qkv, rows, d, 3 * d, name + "qkv")
                k_out, v_out = (b.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
                kv_offset = member * h * (prompt_rows + 1) * hd
                k_out = Tensor(k_out.offset + kv_offset, k_out.shape)
                v_out = Tensor(v_out.offset + kv_offset, v_out.shape)
                b.attention(
                    qkv,
                    ctx,
                    k_out,
                    v_out,
                    rows,
                    0 if phase == "p" else prompt_rows,
                    d,
                    h,
                    hd,
                    name + "a",
                )
                b.gemm(ctx, weights["wo"], att, rows, d, d, name + "wo")
                b.add_rows(x, att, res, rows, d, name + "res")
                b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, rows, d, name + "ln2")
                b.gemm(ln2, weights["w1"], ff1, rows, d, f, name + "w1")
                b.add_bias(ff1, weights["b1"], biased, rows, f, name + "bias1")
                b.gelu(biased, activated, rows, f, name + "gelu")
                b.gemm(activated, weights["w2"], ff2, rows, f, d, name + "w2")
                b.add_rows(res, ff2, y, rows, d, name + "out", weights["b2"])
                x = y
            with b.loop(f"{phase}b{member}output", 0, rows) as row:
                b.ld(hbm(add(x.offset, mul(row, d)), d), rf(0, d))
                b.st(
                    rf(0, d),
                    hbm(
                        add(
                            output.offset,
                            member * (prompt_rows + 1) * d,
                            (0 if phase == "p" else prompt_rows) * d,
                            mul(row, d),
                        ),
                        d,
                    ),
                )
        if phase == "p":
            b.emit("STEP.COMMIT", step=0)
    program = b.finish()
    return program, layout, b.cursor


def generate_m1_d1(gemm_n_tile: int = 16) -> tuple[str, Layout, int]:
    """Compile the D1 model's history and causally released decode steps."""
    model = MODELS["M2"]
    batch, past, steps = SCENARIOS["D1"]
    if batch != 1:
        raise ValueError("D1 baseline requires batch one")
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    layout = build_layout(model, "D1")
    b = Builder(layout, gemm_n_tile=gemm_n_tile)
    output = b.symbol("output/hidden")
    for step in range(steps):
        x = b.symbol(f"input/step{step}")
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            name = f"d{step}l{layer}"
            weights = {
                key: b.symbol(prefix + key)
                for key in (
                    "ln1_g",
                    "ln1_b",
                    "wqkv",
                    "wo",
                    "ln2_g",
                    "ln2_b",
                    "w1",
                    "b1",
                    "w2",
                    "b2",
                )
            }
            ln1 = b.alloc(f"{name}_ln1", 1, d)
            qkv = b.alloc(f"{name}_qkv", 1, 3 * d)
            ctx = b.alloc(f"{name}_ctx", 1, d)
            att = b.alloc(f"{name}_att", 1, d)
            res = b.alloc(f"{name}_res", 1, d)
            ln2 = b.alloc(f"{name}_ln2", 1, d)
            ff1 = b.alloc(f"{name}_ff1", 1, f)
            biased = b.alloc(f"{name}_biased", 1, f)
            activated = b.alloc(f"{name}_gelu", 1, f)
            ff2 = b.alloc(f"{name}_ff2", 1, d)
            y = b.alloc(f"{name}_out", 1, d)
            b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, 1, d, name + "ln1")
            b.gemm(ln1, weights["wqkv"], qkv, 1, d, 3 * d, name + "qkv")
            b.attention_decode(
                qkv,
                ctx,
                b.symbol(prefix + "history_k"),
                b.symbol(prefix + "history_v"),
                b.symbol(prefix + "new_k"),
                b.symbol(prefix + "new_v"),
                step,
                past,
                d,
                h,
                hd,
                name + "a",
            )
            b.gemm(ctx, weights["wo"], att, 1, d, d, name + "wo")
            b.add_rows(x, att, res, 1, d, name + "res")
            b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, 1, d, name + "ln2")
            b.gemm(ln2, weights["w1"], ff1, 1, d, f, name + "w1")
            b.add_bias(ff1, weights["b1"], biased, 1, f, name + "bias1")
            b.gelu(biased, activated, 1, f, name + "gelu")
            b.gemm(activated, weights["w2"], ff2, 1, f, d, name + "w2")
            b.add_rows(res, ff2, y, 1, d, name + "out", weights["b2"])
            x = y
        b.ld(hbm(x.offset, d), rf(0, d))
        b.st(rf(0, d), hbm(output.offset + step * d, d))
        b.emit("STEP.COMMIT", step=step)
    return b.finish(), layout, b.cursor
