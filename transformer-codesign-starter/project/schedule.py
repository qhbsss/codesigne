"""Split both programs across the eight SMs.

P1 runs its two batch elements at the same time, and each SM hosts one
workgroup from each element. Heads and GEMM columns are partitioned across the
eight workers of one element. Prompt attention reuses each K/V tile across a
block of query rows.
D1's eight decode steps stay ordered by STEP.COMMIT. Sixteen workers, two per
SM, give every head four disjoint key ranges.
"""

import json

from codesign.challenge.abi import Layout, build_layout
from codesign.challenge.workload import MODELS, SCENARIOS

from .compiler import Builder, Scratch, Tensor, columns, emit_barrier, heads_for

P1_WORKERS = 8
D1_WORKERS = 16
D1_SEGMENTS = 4
SM_COUNT = 8
BATCH_STRIDE = 40_000_000
SHARED_GUARD = 16_000_000
PRIVATE_CHUNK = 2_000_000
WEIGHT_KEYS = ("ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b", "w1", "b1", "w2", "b2")


def _weights(worker: Builder, prefix: str) -> dict[str, Tensor]:
    return {key: worker.symbol(prefix + key) for key in WEIGHT_KEYS}


def _open_workers(layout: Layout, lines: list[str], prefix: str, origin: int, sms: list[int]):
    workers = []
    for index, sm in enumerate(sms):
        private = Scratch(origin + SHARED_GUARD + index * PRIVATE_CHUNK)
        workers.append(
            Builder(
                layout,
                wg=f"{prefix}{index}",
                sm=sm,
                memory=private,
                lines=lines,
            )
        )
    return workers, Scratch(origin)


def _parts_for(width: int, workers: int, tile: int = 16) -> int:
    """Largest worker count whose column slice is a multiple of the MMA tile."""
    parts = workers
    while parts > 1 and (width % parts or (width // parts) % tile):
        parts //= 2
    if width % parts or (width // parts) % tile:
        raise ValueError(f"{width} cannot be split across {workers} workers")
    return parts


def _decode_assignment(index: int, heads: int, past: int):
    """Four workers share a head and a disjoint quarter of the history keys."""
    if past % D1_SEGMENTS:
        raise ValueError("Decode history is not divisible by the key split")
    head, segment = divmod(index, D1_SEGMENTS)
    if head >= heads:
        raise ValueError("Decode worker is outside the head groups")
    width = past // D1_SEGMENTS
    return head, segment * width, (segment + 1) * width, segment == 0, segment


def _commit(lines: list[str], step: int) -> None:
    lines.append("STEP.COMMIT " + json.dumps({"step": step}, separators=(",", ":")))


def emit_decoder_layer(
    lines: list[str],
    workers: list[Builder],
    shared: Scratch,
    x: Tensor,
    weights: dict[str, Tensor],
    rows: int,
    d: int,
    f: int,
    h: int,
    hd: int,
    name: str,
    *,
    past: int,
    k_out: Tensor | None = None,
    v_out: Tensor | None = None,
    history_k: Tensor | None = None,
    history_v: Tensor | None = None,
    new_k: Tensor | None = None,
    new_v: Tensor | None = None,
    decode_step: int | None = None,
) -> Tensor:
    qkv = shared.alloc(rows, 3 * d)
    ctx = shared.alloc(rows, d)
    res = shared.alloc(rows, d)
    activated = shared.alloc(rows, f)
    y = shared.alloc(rows, d)
    ln1 = shared.alloc(rows, d)
    ln2 = shared.alloc(rows, d)
    parts = len(workers)
    workers[0].layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, rows, d, name + "ln1")
    emit_barrier(lines, workers)
    if decode_step is None:
        for index, worker in enumerate(workers):
            owned = heads_for(h, parts, index)
            if owned is None:
                continue
            head_lo, head_hi = owned
            for component, tag in ((0, "q"), (1, "k"), (2, "v")):
                worker.gemm(
                    ln1,
                    weights["wqkv"],
                    qkv,
                    rows,
                    d,
                    3 * d,
                    name + tag,
                    n_lo=component * d + head_lo * hd,
                    n_hi=component * d + head_hi * hd,
                )
            worker.attention(
                qkv,
                ctx,
                k_out,
                v_out,
                rows,
                past,
                d,
                h,
                hd,
                name + "a",
                head_lo=head_lo,
                head_hi=head_hi,
            )
    else:
        if parts != h * D1_SEGMENTS:
            raise ValueError("Decode schedule requires four workers per head")
        generated = decode_step + 1
        total = past + generated
        scores = shared.alloc(h, total)
        partial = shared.alloc(h, D1_SEGMENTS, hd)
        partial_new = shared.alloc(h, hd)
        for index, worker in enumerate(workers):
            head, _key_lo, _key_hi, primary, _segment = _decode_assignment(index, h, past)
            if not primary:
                continue
            for component, tag in ((0, "q"), (1, "k"), (2, "v")):
                worker.gemm(
                    ln1,
                    weights["wqkv"],
                    qkv,
                    rows,
                    d,
                    3 * d,
                    name + tag,
                    n_lo=component * d + head * hd,
                    n_hi=component * d + (head + 1) * hd,
                )
            worker.export_decode_kv(qkv, new_k, new_v, decode_step, d, head, hd)
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, key_lo, key_hi, primary, _segment = _decode_assignment(index, h, past)
            score_base = scores.offset + head * total
            worker.score_kv_range(
                qkv,
                history_k,
                past,
                score_base,
                head,
                key_lo,
                key_hi,
                0,
                hd,
                name + "hk",
            )
            if primary:
                worker.score_kv_range(
                    qkv,
                    new_k,
                    new_k.shape[2],
                    score_base,
                    head,
                    0,
                    generated,
                    past,
                    hd,
                    name + "nk",
                )
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, _key_lo, _key_hi, primary, _segment = _decode_assignment(index, h, past)
            if primary:
                worker.softmax_scores(scores.offset + head * total, total)
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, key_lo, key_hi, primary, segment = _decode_assignment(index, h, past)
            score_base = scores.offset + head * total
            worker.context_kv_range(
                history_v,
                past,
                score_base,
                partial.offset + (head * D1_SEGMENTS + segment) * hd,
                head,
                key_lo,
                key_hi,
                0,
                hd,
                name + "hv",
            )
            if primary:
                worker.context_kv_range(
                    new_v,
                    new_v.shape[2],
                    score_base,
                    partial_new.offset + head * hd,
                    head,
                    0,
                    generated,
                    past,
                    hd,
                    name + "nv",
                )
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, _key_lo, _key_hi, primary, _segment = _decode_assignment(index, h, past)
            if not primary:
                continue
            worker.combine_context(
                [partial.offset + (head * D1_SEGMENTS + seg) * hd for seg in range(D1_SEGMENTS)]
                + [partial_new.offset + head * hd],
                ctx.offset + head * hd,
                hd,
            )
    emit_barrier(lines, workers)
    d_parts = _parts_for(d, parts)
    for index, worker in enumerate(workers):
        if index >= d_parts:
            continue
        col_lo, col_hi = columns(d, d_parts, index)
        worker.gemm(
            ctx,
            weights["wo"],
            res,
            rows,
            d,
            d,
            name + "wo",
            n_lo=col_lo,
            n_hi=col_hi,
            epilogue="residual",
            residual=x,
        )
    emit_barrier(lines, workers)
    workers[0].layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, rows, d, name + "ln2")
    emit_barrier(lines, workers)
    f_parts = _parts_for(f, parts)
    for index, worker in enumerate(workers):
        if index >= f_parts:
            continue
        col_lo, col_hi = columns(f, f_parts, index)
        worker.gemm(
            ln2,
            weights["w1"],
            activated,
            rows,
            d,
            f,
            name + "w1",
            n_lo=col_lo,
            n_hi=col_hi,
            epilogue="bias_gelu",
            bias=weights["b1"],
        )
    emit_barrier(lines, workers)
    for index, worker in enumerate(workers):
        if index >= d_parts:
            continue
        col_lo, col_hi = columns(d, d_parts, index)
        worker.gemm(
            activated,
            weights["w2"],
            y,
            rows,
            f,
            d,
            name + "w2",
            n_lo=col_lo,
            n_hi=col_hi,
            epilogue="residual_bias",
            residual=res,
            bias=weights["b2"],
        )
    emit_barrier(lines, workers)
    return y


def _guard(shared: Scratch, origin: int) -> None:
    if shared.cursor - origin > SHARED_GUARD:
        raise ValueError("Shared scratch collided with private scratch")


def generate_m1_p1() -> tuple[str, Layout, int]:
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    if P1_WORKERS != SM_COUNT:
        raise ValueError("P1 places one worker from each batch on every SM")
    layout = build_layout(model, "P1")
    lines: list[str] = []
    scratch0 = layout.symbols["scratch"].address // 4
    groups = []
    for member in range(batch):
        origin = scratch0 + member * BATCH_STRIDE
        workers, shared = _open_workers(layout, lines, f"m{member}w", origin, list(range(P1_WORKERS)))
        groups.append((member, workers, shared, origin))
    probe = groups[0][1][0]
    prompt = probe.symbol("input/prompt")
    step = probe.symbol("input/step0")
    output = probe.symbol("output/hidden")
    for phase, rows, past in (("p", prompt_rows, 0), ("s", 1, prompt_rows)):
        for member, workers, shared, origin in groups:
            if phase == "p":
                x = Tensor(prompt.offset + member * prompt_rows * d, (rows, d))
            else:
                x = Tensor(step.offset + member * d, (rows, d))
            for layer in range(model.layers):
                prefix = f"layer{layer}/"
                k_out, v_out = (probe.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
                kv_offset = member * h * (prompt_rows + 1) * hd
                x = emit_decoder_layer(
                    lines,
                    workers,
                    shared,
                    x,
                    _weights(probe, prefix),
                    rows,
                    d,
                    f,
                    h,
                    hd,
                    f"{phase}b{member}l{layer}",
                    past=past,
                    k_out=Tensor(k_out.offset + kv_offset, k_out.shape),
                    v_out=Tensor(v_out.offset + kv_offset, v_out.shape),
                )
            base = output.offset + member * (prompt_rows + 1) * d
            if phase != "p":
                base += prompt_rows * d
            for index, worker in enumerate(workers):
                col_lo, col_hi = columns(d, len(workers), index)
                worker.store_slice(x, base, rows, d, col_lo, col_hi, f"{phase}b{member}out")
            _guard(shared, origin)
        if phase == "p":
            _commit(lines, 0)
    for _, workers, _, _ in groups:
        for worker in workers:
            worker.finish()
    return "\n".join(lines) + "\n", layout, groups[-1][2].cursor


def generate_m1_d1(gemm_n_tile: int = 16) -> tuple[str, Layout, int]:
    """Compile eight decode steps, with each step spread across D1_WORKERS SMs."""
    del gemm_n_tile
    model = MODELS["M2"]
    batch, past, steps = SCENARIOS["D1"]
    if batch != 1:
        raise ValueError("D1 requires batch one")
    if D1_WORKERS != SM_COUNT * 2:
        raise ValueError("D1 places two workers on every SM")
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    layout = build_layout(model, "D1")
    lines: list[str] = []
    scratch0 = layout.symbols["scratch"].address // 4
    workers, shared = _open_workers(
        layout, lines, "d", scratch0, [index % SM_COUNT for index in range(D1_WORKERS)]
    )
    probe = workers[0]
    output = probe.symbol("output/hidden")
    for step in range(steps):
        x = probe.symbol(f"input/step{step}")
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            x = emit_decoder_layer(
                lines,
                workers,
                shared,
                x,
                _weights(probe, prefix),
                1,
                d,
                f,
                h,
                hd,
                f"s{step}l{layer}",
                past=past,
                history_k=probe.symbol(prefix + "history_k"),
                history_v=probe.symbol(prefix + "history_v"),
                new_k=probe.symbol(prefix + "new_k"),
                new_v=probe.symbol(prefix + "new_v"),
                decode_step=step,
            )
        out_parts = _parts_for(d, len(workers))
        for index, worker in enumerate(workers):
            if index >= out_parts:
                continue
            col_lo, col_hi = columns(d, out_parts, index)
            worker.store_slice(x, output.offset + step * d, 1, d, col_lo, col_hi, f"s{step}out")
        _commit(lines, step)
    _guard(shared, scratch0)
    for worker in workers:
        worker.finish()
    return "\n".join(lines) + "\n", layout, shared.cursor
