"""Split both programs across the eight SMs.

P1 runs its two batch elements at the same time, and each SM hosts one
workgroup from each element. Heads and GEMM columns are partitioned across the
eight workers of one element. Prompt attention reuses each K/V tile across a
block of query rows.
D1's eight decode steps stay ordered by STEP.COMMIT. Thirty-two workers, two
per SM, give every head eight disjoint key ranges.
"""

import json

from codesign.challenge.abi import Layout, build_layout
from codesign.challenge.workload import MODELS, SCENARIOS

from .compiler import Builder, Scratch, Tensor, columns, emit_barrier, heads_for
from .compact_combine import combine
from .block_epilogue import reduce_four
Builder.reduce_w2_four_n4 = reduce_four
Builder.combine_online_attention_persistent = combine

P1_WORKERS = 16
D1_WORKERS = 32
D1_SEGMENTS = 8
SM_COUNT = 16
P1_SM_COUNT = 16
P1_SHARED_BYTES_PER_WORKER = 0
D1_HIGH_SMS = [(index + 4) % SM_COUNT for index in range(SM_COUNT)]
D1_HIGH_SMS[4], D1_HIGH_SMS[13] = D1_HIGH_SMS[13], D1_HIGH_SMS[4]
D1_SMS = list(range(SM_COUNT)) + D1_HIGH_SMS
D1_QKV_WORKERS = list(range(16)) + list(range(17, 24)) + [25]
D1_QKV_TASK = {worker: task for task, worker in enumerate(D1_QKV_WORKERS)}
D1_SHARED_BYTES_PER_WORKER = 0
BATCH_STRIDE = 40_000_000
SHARED_GUARD = 16_000_000
PRIVATE_CHUNK = 2_000_000
WEIGHT_KEYS = ("ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b", "w1", "b1", "w2", "b2")


def _weights(worker: Builder, prefix: str) -> dict[str, Tensor]:
    return {key: worker.symbol(prefix + key) for key in WEIGHT_KEYS}


def _open_workers(
    layout: Layout,
    lines: list[str],
    prefix: str,
    origin: int,
    sms: list[int],
    *,
    shared_bytes: int = 0,
):
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
                shared_bytes=shared_bytes,
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


def _align_p1_prompt_stages(lines: list[str]) -> list[str]:
    """Run the two prompt batches behind one barrier per matching stage.

    The timed cache intentionally does not merge pending misses.  With two
    independent local barriers, tiny timing drift makes both batches miss on
    the same weight lines.  Rebuilding the prompt prefix stage-by-stage gives
    corresponding weight loads the same issue cycle, allowing the hardware's
    multicast path to serve both destinations from one source request.

    The first STEP.COMMIT remains in its original place and the fused step
    suffix is copied verbatim.  Only the order of independent prompt-batch
    instructions and their synchronization scope changes.
    """
    commit_index = next(
        (index for index, line in enumerate(lines) if line.startswith("STEP.COMMIT ")),
        None,
    )
    if commit_index is None:
        raise ValueError("P1 alignment requires a prompt commit")
    prefix = lines[:commit_index]
    suffix = lines[commit_index:]
    begins = [line for line in prefix if line.startswith("WG.BEGIN ")]
    buffers = {0: [], 1: []}
    stages = {0: [], 1: []}
    current_member = None

    for line in prefix:
        if line.startswith("WG.BEGIN "):
            continue
        if line.startswith("BARRIER "):
            payload = json.loads(line.split(" ", 1)[1])
            wgs = payload["wgs"]
            if wgs and all(wg.startswith("m0w") for wg in wgs):
                member = 0
            elif wgs and all(wg.startswith("m1w") for wg in wgs):
                member = 1
            else:
                raise ValueError("Unexpected pre-commit barrier scope")
            stages[member].append(buffers[member])
            buffers[member] = []
            continue
        in0 = '"m0w' in line
        in1 = '"m1w' in line
        if not in0 and not in1 and line.startswith("END.FOR ") and current_member is not None:
            buffers[current_member].append(line)
            continue
        if in0 == in1:
            raise ValueError(f"Cannot assign P1 prompt instruction to one batch: {line[:200]}")
        current_member = 1 if in1 else 0
        buffers[current_member].append(line)

    if len(stages[0]) != len(stages[1]) or not stages[0]:
        raise ValueError("Unexpected P1 prompt stage count")
    all_wgs = [f"m{member}w{worker}" for member in range(2) for worker in range(P1_WORKERS)]
    barrier = "BARRIER " + json.dumps({"wgs": all_wgs, "events": []}, separators=(",", ":"))
    rebuilt = list(begins)
    for left, right in zip(stages[0], stages[1]):
        rebuilt.extend(left)
        rebuilt.extend(right)
        rebuilt.append(barrier)
    rebuilt.extend(buffers[0])
    rebuilt.extend(buffers[1])
    rebuilt.extend(suffix)
    return rebuilt


def emit_decoder_layer(
    lines: list[str],
    workers: list[Builder],
    shared: Scratch,
    x: Tensor,
    weights: dict[str, Tensor] | list[dict[str, Tensor]],
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
    def weight(index: int, key: str) -> Tensor:
        return weights[index][key] if isinstance(weights, list) else weights[key]

    qkv = shared.alloc(rows, 3 * d)
    ctx = shared.alloc(rows, d)
    res = shared.alloc(rows, d)
    activated = shared.alloc(rows, f)
    y = shared.alloc(rows, d)
    parts = len(workers)
    w2_tasks = 16 if decode_step is None and rows > 1 and parts == 16 else 8
    w2_partial = shared.alloc(w2_tasks, rows, d) if decode_step is None and rows > 1 else None
    ln1 = shared.alloc(rows, d)
    ln2 = shared.alloc(rows, d)
    if decode_step is None and rows >= len(workers):
        for index, worker in enumerate(workers):
            row_lo, row_hi = columns(rows, len(workers), index)
            worker.layernorm(
                x,
                weight(index, "ln1_g"),
                weight(index, "ln1_b"),
                ln1,
                rows,
                d,
                name + "ln1",
                row_lo,
                row_hi,
            )
    else:
        if decode_step is not None and isinstance(weights, list):
            workers[0].layernorm_persistent(
                x, weight(0, "ln1_g"), weight(0, "ln1_b"), ln1, d
            )
        else:
            workers[0].layernorm(
                x, weight(0, "ln1_g"), weight(0, "ln1_b"), ln1, rows, d, name + "ln1"
            )
    emit_barrier(lines, workers)
    if decode_step is None:
        if parts == 16 and h == 8 and rows == 64 and past == 0:
            # Two workers per head.  Each owns 32 rows, so its private 16 KiB
            # SH quota is no longer needed for a full 64-row QKV head.  QKV is
            # materialized in HBM, then the two query-block halves consume the
            # shared head-major K/V exports independently.
            half_rows = rows // 2
            for index, worker in enumerate(workers):
                head, half = divmod(index, 2)
                row_lo = half * half_rows
                worker.gemm_reuse_a(
                    Tensor(ln1.offset + row_lo * d, (half_rows, d), row_stride=d),
                    weight(index, "wqkv"),
                    Tensor(qkv.offset + row_lo * 3 * d, (half_rows, 3 * d), row_stride=3 * d),
                    half_rows,
                    d,
                    3 * d,
                    name + f"qkv{index}",
                    tuple(
                        (component * d + head * hd, component * d + (head + 1) * hd)
                        for component in range(3)
                    ),
                )
            emit_barrier(lines, workers)
            for index, worker in enumerate(workers):
                head, half = divmod(index, 2)
                row_lo = half * half_rows
                worker.export_prompt_kv_rows(
                    qkv, k_out, v_out, row_lo, row_lo + half_rows,
                    d, head, hd, name + f"x{index}",
                )
            emit_barrier(lines, workers)
            for index, worker in enumerate(workers):
                head, half = divmod(index, 2)
                worker.attention_prompt_slice(
                    qkv, ctx, k_out, v_out, rows, d, hd,
                    name + f"a{index}", head, half * 2, half * 2 + 2,
                )
        else:
            for index, worker in enumerate(workers):
                owned = heads_for(h, parts, index)
                if owned is None:
                    continue
                head_lo, head_hi = owned
                if head_hi - head_lo != 1:
                    raise ValueError("P1 QKV A-reuse expects one head per worker")
                qkv_worker = (
                    Tensor(0, (rows, 3 * hd), space="SH", row_stride=3 * hd)
                    if rows > 1
                    else qkv
                )
                worker.gemm_reuse_a(
                    ln1,
                    weight(index, "wqkv"),
                    qkv_worker,
                    rows,
                    d,
                    3 * d,
                    name + "qkv",
                    tuple(
                        (component * d + head_lo * hd, component * d + head_hi * hd)
                        for component in range(3)
                    ),
                    out_column_ranges=(
                        ((0, hd), (hd, 2 * hd), (2 * hd, 3 * hd))
                        if qkv_worker.space == "SH"
                        else None
                    ),
                )
                worker.attention(
                    qkv_worker,
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
            raise ValueError("Decode schedule worker/head partition mismatch")
        generated = decode_step + 1
        total = past + generated
        partial_max = shared.alloc(h, D1_SEGMENTS)
        partial_sum = shared.alloc(h, D1_SEGMENTS)
        partial = shared.alloc(h, D1_SEGMENTS, hd)
        for index, worker in enumerate(workers):
            if index not in D1_QKV_TASK:
                continue
            component, head_half = divmod(D1_QKV_TASK[index], 8)
            head, half = divmod(head_half, 2)
            qkv_lo = component * d + head * hd + half * 16
            worker.gemm(
                ln1,
                weight(index, "wqkv"),
                qkv,
                rows,
                d,
                3 * d,
                name + ("q", "k", "v")[component],
                n_lo=qkv_lo,
                n_hi=qkv_lo + 16,
            )
            if component == 1:
                worker.export_decode_component(
                    qkv, new_k, decode_step, d, head, hd, component, half * 16, 16
                )
            elif component == 2:
                worker.export_decode_component(
                    qkv, new_v, decode_step, d, head, hd, component, half * 16, 16
                )
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, key_lo, key_hi, primary, segment = _decode_assignment(index, h, past)
            worker.online_attention_local_persistent(
                qkv,
                history_k,
                history_v,
                new_k,
                new_v,
                head,
                key_lo,
                key_hi,
                generated,
                primary,
                hd,
                partial_max.offset + head * D1_SEGMENTS + segment,
                partial_sum.offset + head * D1_SEGMENTS + segment,
                partial.offset + (head * D1_SEGMENTS + segment) * hd,
            )
        emit_barrier(lines, workers)
        for index, worker in enumerate(workers):
            head, _key_lo, _key_hi, primary, _segment = _decode_assignment(index, h, past)
            if not primary:
                continue
            worker.combine_online_attention_persistent(
                partial_max.offset + head * D1_SEGMENTS,
                partial_sum.offset + head * D1_SEGMENTS,
                partial.offset + head * D1_SEGMENTS * hd,
                ctx.offset + head * hd,
                D1_SEGMENTS,
                hd,
            )
    emit_barrier(lines, workers)
    d_parts = 16 if decode_step is not None else _parts_for(d, parts, tile=16)
    for index, worker in enumerate(workers):
        if index >= d_parts:
            continue
        col_lo, col_hi = columns(d, d_parts, index)
        worker.gemm(
            ctx,
            weight(index, "wo"),
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
    if decode_step is None and rows >= len(workers):
        for index, worker in enumerate(workers):
            row_lo, row_hi = columns(rows, len(workers), index)
            worker.layernorm(
                res,
                weight(index, "ln2_g"),
                weight(index, "ln2_b"),
                ln2,
                rows,
                d,
                name + "ln2",
                row_lo,
                row_hi,
            )
    else:
        if decode_step is not None and isinstance(weights, list):
            workers[0].layernorm_persistent(
                res, weight(0, "ln2_g"), weight(0, "ln2_b"), ln2, d
            )
        else:
            workers[0].layernorm(
                res, weight(0, "ln2_g"), weight(0, "ln2_b"), ln2, rows, d, name + "ln2"
            )
    emit_barrier(lines, workers)
    f_parts = 16 if decode_step is not None else _parts_for(f, parts)
    f_workers = (
        [(index, workers[index]) for index in range(parts)]
        if decode_step is None
        else [(index, workers[index]) for index in range(16, 32)]
    )
    for index, worker in f_workers:
        part_index = index if decode_step is None else index - 16
        if part_index >= f_parts:
            continue
        col_lo, col_hi = columns(f, f_parts, part_index)
        if decode_step is None and col_hi - col_lo == 128:
            worker.gemm_reuse_a(
                ln2,
                weight(index, "w1"),
                activated,
                rows,
                d,
                f,
                name + "w1r",
                ((col_lo, col_lo + 64), (col_lo + 64, col_hi)),
                epilogue="bias_gelu",
                bias=weight(index, "b1"),
            )
        else:
            worker.gemm(
                ln2,
                weight(index, "w1"),
                activated,
                rows,
                d,
                f,
                name + "w1",
                n_lo=col_lo,
                n_hi=col_hi,
                epilogue="bias_gelu",
                bias=weight(index, "b1"),
            )
    emit_barrier(lines, workers)
    if w2_partial is not None:
        if parts == 16:
            # Four K slices x four N slices.  Relative to the 8-worker 4x2
            # schedule, every task has half the output width while keeping the
            # same four-way reduction tree.
            k_width = f // 4
            n_width = d // 4
            for index, worker in enumerate(workers):
                k_group, n_group = divmod(index, 4)
                k_lo = k_group * k_width
                n_lo = n_group * n_width
                worker.gemm(
                    Tensor(activated.offset + k_lo, (rows, k_width), row_stride=f),
                    Tensor(weight(index, "w2").offset + k_lo * d, (k_width, d)),
                    Tensor(w2_partial.offset + index * rows * d, (rows, d)),
                    rows,
                    k_width,
                    d,
                    name + f"w2p{index}",
                    n_lo=n_lo,
                    n_hi=n_lo + n_width,
                )
            emit_barrier(lines, workers)
            for index, worker in enumerate(workers):
                row_group, n_group = divmod(index, 4)
                row_lo, row_hi = columns(rows, 4, row_group)
                worker.reduce_w2_four_n4(
                    w2_partial, y, res, weight(index, "b2"), rows, d,
                    n_group, row_lo, row_hi, name + f"w2r{index}",
                )
        else:
            # 4 K slices x 2 N slices: W1 activations are read twice rather than
            # eight times.  Each task has the same k*n product as the old 1x8
            # column partition, and two N tiles share each activation load.
            k_width = f // 4
            n_width = d // 2
            for index, worker in enumerate(workers):
                k_group, n_group = divmod(index, 2)
                k_lo = k_group * k_width
                n_lo = n_group * n_width
                worker.gemm_reuse_a(
                    Tensor(activated.offset + k_lo, (rows, k_width), row_stride=f),
                    Tensor(weight(index, "w2").offset + k_lo * d, (k_width, d)),
                    Tensor(w2_partial.offset + index * rows * d, (rows, d)),
                    rows,
                    k_width,
                    d,
                    name + f"w2p{index}",
                    ((n_lo, n_lo + 64), (n_lo + 64, n_lo + n_width)),
                )
            emit_barrier(lines, workers)
            row_parts = 4
            for index, worker in enumerate(workers):
                row_group, n_group = divmod(index, 2)
                row_lo, row_hi = columns(rows, row_parts, row_group)
                n_lo, n_hi = columns(d, 2, n_group)
                worker.reduce_w2_four(
                    w2_partial, y, res, weight(index, "b2"), rows, d,
                    n_lo, n_hi, row_lo, row_hi, name + f"w2r{index}",
                )
    else:
        for index, worker in enumerate(workers):
            if index >= d_parts:
                continue
            col_lo, col_hi = columns(d, d_parts, index)
            worker.gemm(
                activated,
                weight(index, "w2"),
                y,
                rows,
                f,
                d,
                name + "w2",
                n_lo=col_lo,
                n_hi=col_hi,
                epilogue="residual_bias",
                residual=res,
                bias=weight(index, "b2"),
            )
    emit_barrier(lines, workers)
    return y


def _guard(shared: Scratch, origin: int) -> None:
    if shared.cursor - origin > SHARED_GUARD:
        raise ValueError("Shared scratch collided with private scratch")


from .parallel_step import emit_p1_fused_step_layer

def generate_m1_p1() -> tuple[str, Layout, int]:
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    if batch != 2 or P1_WORKERS != 16 or P1_SM_COUNT != 16:
        raise ValueError("P1 two-resident-worker schedule shape changed")
    layout = build_layout(model, "P1")
    lines: list[str] = []
    scratch0 = layout.symbols["scratch"].address // 4
    groups = []
    for member in range(batch):
        origin = scratch0 + member * BATCH_STRIDE
        sm_lo = member * (P1_SM_COUNT // batch)
        workers, shared = _open_workers(
            layout,
            lines,
            f"m{member}w",
            origin,
            [sm_lo + index % (P1_SM_COUNT // batch) for index in range(P1_WORKERS)],
            shared_bytes=P1_SHARED_BYTES_PER_WORKER,
        )
        groups.append((member, workers, shared, origin))
    probe = groups[0][1][0]
    prompt = probe.symbol("input/prompt")
    step = probe.symbol("input/step0")
    output = probe.symbol("output/hidden")
    for member, workers, shared, origin in groups:
        x = Tensor(prompt.offset + member * prompt_rows * d, (prompt_rows, d))
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            k_out, v_out = (probe.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
            kv_offset = member * h * (prompt_rows + 1) * hd
            x = emit_decoder_layer(
                lines, workers, shared, x, _weights(probe, prefix),
                prompt_rows, d, f, h, hd, f"pb{member}l{layer}",
                past=0,
                k_out=Tensor(k_out.offset + kv_offset, k_out.shape),
                v_out=Tensor(v_out.offset + kv_offset, v_out.shape),
            )
        base = output.offset + member * (prompt_rows + 1) * d
        for index, worker in enumerate(workers):
            col_lo, col_hi = columns(d, len(workers), index)
            worker.store_slice(x, base, prompt_rows, d, col_lo, col_hi, f"pb{member}out")
        _guard(shared, origin)
    _commit(lines, 0)

    workers, shared, origin = groups[0][1] + groups[1][1], groups[0][2], groups[0][3]
    x = Tensor(step.offset, (batch, d))
    for layer in range(model.layers):
        prefix = f"layer{layer}/"
        base_k, base_v = (probe.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
        stride = h * (prompt_rows + 1) * hd
        x = emit_p1_fused_step_layer(
            lines,
            workers,
            shared,
            x,
            _weights(probe, prefix),
            d,
            f,
            h,
            hd,
            f"sfl{layer}",
            prompt_rows,
            [Tensor(base_k.offset + member * stride, base_k.shape) for member in range(batch)],
            [Tensor(base_v.offset + member * stride, base_v.shape) for member in range(batch)],
        )
    for member in range(batch):
        base = output.offset + (member * (prompt_rows + 1) + prompt_rows) * d
        row = Tensor(x.offset + member * d, (1, d))
        for index, worker in enumerate(workers):
            col_lo, col_hi = columns(d, len(workers), index)
            worker.store_slice(row, base, 1, d, col_lo, col_hi, f"sfb{member}out")
    _guard(shared, origin)
    for _, workers, _, _ in groups:
        for worker in workers:
            worker.finish()
    lines = _align_p1_prompt_stages(lines)
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
        layout,
        lines,
        "d",
        scratch0,
        D1_SMS,
        shared_bytes=D1_SHARED_BYTES_PER_WORKER,
    )
    probe = workers[0]
    output = probe.symbol("output/hidden")
    cached_layers: list[list[dict[str, Tensor]]] = []
    for layer in range(model.layers):
        base = _weights(probe, f"layer{layer}/")
        per_worker = []
        for index, worker in enumerate(workers):
            item = dict(base)
            if index in D1_QKV_TASK:
                component, head_half = divmod(D1_QKV_TASK[index], 8)
                head, half = divmod(head_half, 2)
                qkv_lo = component * d + head * hd + half * 16
                item["wqkv"] = worker.cache_weight_rf(
                    base["wqkv"], qkv_lo, qkv_lo + 16
                )
            if index < 16:
                d_lo, d_hi = columns(d, 16, index)
                item["wo"] = worker.cache_weight_rf(base["wo"], d_lo, d_hi)
                item["w2"] = worker.cache_weight_rf(base["w2"], d_lo, d_hi)
            else:
                f_lo, f_hi = columns(f, 16, index - 16)
                item["w1"] = worker.cache_weight_rf(base["w1"], f_lo, f_hi)
            per_worker.append(item)
        cached_layers.append(per_worker)
    for step in range(steps):
        x = probe.symbol(f"input/step{step}")
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            x = emit_decoder_layer(
                lines,
                workers,
                shared,
                x,
                cached_layers[layer],
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
        out_parts = 16
        for index, worker in enumerate(workers):
            if index >= out_parts:
                continue
            col_lo, col_hi = columns(d, out_parts, index)
            worker.store_slice_persistent(
                x, output.offset + step * d, d, col_lo, col_hi
            )
        _commit(lines, step)
    _guard(shared, scratch0)
    for worker in workers:
        worker.finish()
    return "\n".join(lines) + "\n", layout, shared.cursor

from .rf_prompt import RFBuilder,generate_m1_p1
Builder = RFBuilder
