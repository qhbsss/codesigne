# Phase Two: Memory ABI and allocation

This ABI applies to `phase-two-static-v1`. Values are FP32. All lengths, bases and strides in program records count **FP32 elements**, unless a field explicitly says KiB or bytes. HBM timing addresses are bytes. The evaluator supplies values from its seed; programs bind names rather than embedding input values.

[TOC]

## Canonical inputs

`i` is a zero-based layer index; `s` is the current step. Matrices are row-major. D is hidden width, F is FFN width, B is batch, P is prompt length and T is initial history.

| Source | Shape / elements | Meaning |
| --- | --- | --- |
| `input0` (prefill) | `[B,P,D]` | Entire prompt |
| `input{s}` (decode) | `[B,D]` | Current external decode input, s=0..3 |
| `layer{i}.qkv` | `[D,3D]` | Q, K, V projection columns in that order |
| `layer{i}.qb` | `[3D]` | QKV bias |
| `layer{i}.out`, `.ob` | `[D,D]`, `[D]` | Attention output projection and bias |
| `layer{i}.up`, `.ub` | `[D,F]`, `[F]` | FFN first projection and bias |
| `layer{i}.down`, `.db` | `[F,D]`, `[D]` | FFN second projection and bias |
| `layer{i}.g1`, `.b1`, `.g2`, `.b2` | Each `[D]` | Affine LayerNorm scale/bias |
| `layer{i}.past_k`, `.past_v` | `[T,B,D]` | Decode historical KV only |

Each source may be bound once. Use canonical spellings (`layer0`, not `layer00`). `input{s}` is available only at step s; a successful commit unlocks the next input. Other requests and heads cannot attend to each other.

```json
{"record":"input","source":"input0","len":1536}
{"record":"alloc","len":4096}
```

The first record above is valid for W-D1. Every `input` or `alloc` receives the next zero-based live tensor index. Instructions refer to this index, never to an arbitrary absolute HBM address. For normal inputs, `len` equals the source element count. A historical KV binding may reserve between `T*B*D` and `(T+steps)*B*D` elements so new KV can be appended. Its unused tail must be initialized before reading.

## Placement and lifetime

Each allocation begins at the next 256-byte-aligned HBM address after the previous live allocation. The candidate chooses allocation order and sizes; the simulator chooses the corresponding byte bases. A global view addresses `tensor_base_bytes + 4*(base + row*row_stride + col*col_stride)`. Channels use `(byte_address/256) % hbm_channels`; cache lines are 64 bytes. Allocation order and paid packing can therefore change channel/cache behavior.

Scratch `alloc.len` must be positive and starts uninitialized (NaN). Bound inputs cannot be released. `release` may remove only the last live tensor, after a completed wave; release multiple scratch tensors in reverse order. The index and address may subsequently be reused. Do not retain references to released allocations. Release invalidates that allocation's cached range; it is not a general cache-flush instruction.

```json
{"record":"release","tensor":1}
```

The example releases the scratch tensor above, assuming no later allocation remains. Input bindings and scratch count toward cumulative allocation (64 GiB); current aligned live allocation must fit 2 GiB. Releasing memory does not refund cumulative allocation work.

## Local storage and views

Each wave group reserves `rf_kib` and `sh_kib` on its selected SM. Its local addresses start at zero and are private to that group. Per-SM reservations must fit the hardware capacities and resident-group limit. Groups share the physical RF/SH service ports, banks and engines. RF and SH data do not persist across waves; keep cross-wave values in HBM with paid stores and loads.

A view has `base`, `rows`, `cols`, `row_stride`, `col_stride`. Logical element `(r,c)` addresses `base+r*row_stride+c*col_stride`. A contiguous 2×3 matrix is:

```json
{"base":0,"rows":2,"cols":3,"row_stride":3,"col_stride":1}
```

Zero strides allow read broadcasts. Transfer source and destination must have identical `rows` and `cols`; equal element counts alone do not allow reshape. For HBM↔RF, HBM↔SH and SH↔RF transfers, the destination must have `col_stride = 1` and `row_stride >= cols`, including single-row transfers. Source strides may represent broadcasts or strided reads. General scatter destinations are unsupported; use explicit paid transfers into legal contiguous rows. All accessed elements must lie in the bound tensor or the group's reserved local storage. Transpose is represented by strides or by explicit paid packing, with actual bank and line traffic charged.

## Outputs and step commits

A commit contains `hidden`, `keys`, `values`. Each output view is `{tensor,base,len}` and is contiguous. Keys and values are arrays with exactly one view per layer, in layer order. Each view contains `B*P*D` elements for prefill or `B*D` for a decode step, ordered `[batch,token,D]` (token count is one for decode). Submit the **new** K/V, not the complete historical KV. Historical storage remains `[time,batch,D]`; rearrangements must use paid operations.

```json
{"record":"commit","outputs":{"hidden":{"tensor":20,"base":0,"len":1536},"keys":[{"tensor":21,"base":0,"len":1536},{"tensor":22,"base":0,"len":1536}],"values":[{"tensor":23,"base":0,"len":1536},{"tensor":24,"base":0,"len":1536}]}}
```

These illustrative IDs must be replaced by actual live output tensors. The evaluator checks all outputs immediately against independent reference computation. Prefill commits once; decode commits four times. Nonfinite values fail; tolerance is `1e-3 + 1e-3*abs(reference)`. No records may follow the final commit. A correct commit opens the next external input; it does not compute or copy the next step for you.

See [static ISA](ISA.md) for instructions and [walkthrough](WALKTHROUGH.md) for runnable examples. The complete implementation is in the starter's `source/src/{v09_submission,submission,machine}.rs`.
