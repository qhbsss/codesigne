# Complete assembly reference

This is the public instruction set for the frozen two-case challenge. For the structure of the two complete Transformer programs, see the [assembly walkthrough](/assembly-guide/) and the [P1](/examples/M1_P1.asm) and [D1](/examples/M2_D1.asm) listings. Every nonempty line has the form `OPCODE {"field": ...}`. A line beginning with `#` is a comment. JSON objects must have unique keys and finite numeric values; unknown opcodes and malformed loop bodies are rejected, including bodies of zero-trip loops. Programs are FP32, have no data-dependent branch, and may use at most 8 MiB of source text, 32 nested static loops, and 10 million expanded instructions per case.

**Source of truth:** the starter ZIP contains the exact parser (`codesign/challenge/isa.py`), functional executor (`micro.py`), race checker (`hbm_race.py`), timing scheduler (`pipeline_events.py`), HBM layout (`abi.py`), and local command line checker (`challenge.py`). This page states the complete authoring contract; use those files when checking a corner case. `OPCODE` and JSON must be on the **same physical line**. Blank lines and whole-line `#` comments are accepted. Inline comments, labels, registers named in free-form text, branches, and opcodes outside the table below are not accepted. Operand keys and opcode spellings are case-sensitive. Except for the documented legacy sync shorthand, use exactly the listed top-level fields.

## Operand format

A memory view is a JSON object such as:

```json
{"space":"RF","offset":0,"count":48,"wg":"g0","lane":0}
```

| Field | Meaning |
| --- | --- |
| `space` | `HBM`, `SH`, or `RF` |
| `offset` | Starting FP32 **element** offset within the selected space |
| `count` | Number of logical FP32 elements, positive |
| `wg` | Workgroup name for RF/SH; `null` for HBM |
| `lane` | RF storage partition number; 0 for HBM/SH |
| `shape` (optional) | One- or two-dimensional positive logical extents |
| `strides` (optional) | Matching positive element strides; `count` equals the product of `shape` |

All five base fields are required, even for HBM. `HBM` uses `"wg":null,"lane":0`; `SH` uses the active workgroup's name and lane 0; `RF` uses the active workgroup's name and a lane index. `offset` and `count` are nonnegative/positive **integers**, not byte addresses or booleans. Every word is four bytes. The actual address of a shaped view is `offset + i*strides[0]` for one dimension, or `offset + i*strides[0] + j*strides[1]` for two; the element order is row-major over the logical indices. The largest touched offset must fit the space. For a contiguous view, omit both optional fields. For a one-dimensional strided view, specify `"shape":[count],"strides":[stride]`; for a matrix tile, specify both two-element arrays. A destination view must not touch the same physical location twice.

Without `shape`/`strides`, the view is contiguous. For a two-dimensional tile, supply **both** arrays: `count` is the product of `shape`, while `strides` gives the physical element increments in the source space. A write view cannot visit the same element twice. HBM offsets come from the public [ABI list](/abi/) and have a 2 GiB boundary. RF is private to one workgroup and partition; SH is private to one workgroup. A workgroup's RF slot is 64 KiB, divided among the RF partitions selected by the hardware vector width. SH accesses must stay within its declared byte quota. No instruction may read unwritten RF/SH data, cross into another workgroup's RF/SH, write a read-only HBM symbol, or read a future step's input. The [assembly walkthrough](/assembly-guide/#4-read-hbm-operands-as-element-addresses) works through a real strided weight tile.

For RF, `lane` ranges from 0 through `vector_lanes - 1`; each partition has `65536 / (4 * vector_lanes)` FP32 elements. RF offsets are **within a partition**, so lane 1 offset 0 is a different word from lane 0 offset 0. SH has `shared_bytes / 4` words per workgroup. HBM is a flat `2 GiB / 4`-word space; consult the [public ABI](/abi/) for named ranges. Scratch and required outputs begin invalid and need a write before a read. Required output/KV positions cannot be written twice; weights, original inputs, and history are read-only. A transfer or compute may overwrite its own input only when the operation's read has completed; the scheduler checks pending hazards.

An `event` is a unique nonempty name for an operation. Loop-index interpolation such as `"load_{i}"` is allowed after a `FOR` binds `i`.

**Operand helpers for generators:** a transfer's `src` and `dst` are full memory views. `MMA.ACC` uses three RF views named `a`, `b`, `acc`. `REDUCE` and `SFU` use RF `src` and `dst`. `VEC.src` is an array of RF views or immediate objects of the exact form `{"imm":number}`; `VEC.dst` is an RF view. There is no implicit HBM-to-compute path: load into RF first. A destination is initialized by a prior operation unless the instruction overwrites every destination element without reading it; `MMA.ACC` specifically **reads and updates** `acc`.

## Complete opcode list

| Opcode | Required JSON fields | Effect |
| --- | --- | --- |
| `WG.BEGIN` | `wg`, `sm`, `shared_bytes` | Start a named workgroup on a target SM and reserve a private RF slot and SH quota. Quota is a nonnegative multiple of four bytes. |
| `WG.END` | `wg` | End a workgroup and release its RF/SH contents and residency. |
| `LD` | `src`, `dst`, `event` | Copy an equal-size view between different spaces. HBM↔RF, HBM↔SH, and SH↔RF are allowed. |
| `ST` | `src`, `dst`, `event` | The same explicit transfer semantics as `LD`; commonly used for writes to HBM. |
| `MMA.ACC` | `a`, `b`, `acc`, `m`, `n`, `k`, `event` | Accumulate an `m×k` RF matrix times a `k×n` RF matrix into an `m×n` RF accumulator. Requires TC > 0. |
| `VEC` | `kind`, `src`, `dst`, `event` | Elementwise RF operation. `src` is a list of RF views or finite FP32 immediates. |
| `REDUCE` | `kind`, `src`, `dst`, `event` | Reduce one RF vector with `sum` or `max` to a one-element RF destination. |
| `SFU` | `kind`, `src`, `dst`, `event` | Apply `exp`, `rsqrt`, or `tanh` elementwise to an RF vector. |
| `WAIT` | `wg`, `events` | Make one active workgroup wait for named earlier events. |
| `BARRIER` | `wgs`, `events` | Synchronize the listed active workgroups and named earlier events. |
| `STEP.COMMIT` | `step` | Global step barrier, with zero-based consecutive step numbers. Wait for preceding work and check output/KV coverage before releasing the next external input. |
| `FOR` | `var`, `start`, `stop`, `step` | Start a static half-open integer loop; `step` is positive. |
| `END.FOR` | no fields (`{}`) | End the matching loop. |

### Exact instruction behavior

- **`WG.BEGIN {"wg":string,"sm":int,"shared_bytes":int}`**: `wg` is nonempty and not already active. `sm` is in `[0, sm_count)`. `shared_bytes` is a nonnegative multiple of 4, no larger than that SM's total shared-memory capacity. It creates fresh private RF/SH validity state. At most four workgroups can be physically resident on an SM, and concurrent resident SH quotas must fit; more logical groups may wait in the timing scheduler. **`WG.END {"wg":string}`** closes an active group, waits for its pending transfers, and permits later reuse of the name with fresh storage. Every group must end before program completion.
- **`LD` and `ST`** each take exactly `{"src":view,"dst":view,"event":string}`. They perform the same elementwise copy of equal positive logical counts between **different** spaces: HBM↔RF, HBM↔SH, or SH↔RF. `ST` is conventionally used when writing HBM, but its opcode does not impose direction. RF↔SH requires the same workgroup in both views. Cross-workgroup RF/SH copies and transfers within one space are invalid. The two views may have different shapes/strides if their flattened logical element counts agree; values copy in logical order. An event identifies this transfer for later synchronization.
- **`MMA.ACC {"a":view,"b":view,"acc":view,"m":int,"n":int,"k":int,"event":string}`**: `m,n,k` are positive. All three views are RF in the same workgroup, possibly in different lanes. Flattened `a`, `b`, and `acc` counts must be `m*k`, `k*n`, and `m*n`. Reshape them row-major and compute `acc += a @ b`, writing the accumulator back in place. TC count must be nonzero. Initialize `acc`, commonly with `VEC add` of two zero immediates, before the first MMA. Instruction tile dimensions need not equal physical TC dimensions; the service model charges hardware tile tails.
- **`VEC {"kind":string,"src":[...],"dst":view,"event":string}`**: all memory operands are RF in the same workgroup. Each RF source has either one value, broadcast to every destination element, or exactly the destination count. Immediates broadcast after conversion to finite FP32. Source order and exact arity are defined in the vector table below. Destination must have a positive count; results must be finite when written.
- **`REDUCE {"kind":"sum"|"max","src":view,"dst":view,"event":string}`**: RF source and RF destination belong to the same workgroup; source count is positive and destination count must be **one**. `sum` uses a neighboring-pair FP32 binary tree, carrying an odd final item to the next level. `max` takes the maximum. The single result must be finite.
- **`SFU {"kind":"exp"|"rsqrt"|"tanh","src":view,"dst":view,"event":string}`**: RF source/destination in one workgroup with **equal counts**. The operations are elementwise `exp(x)`, `1/sqrt(x)`, and `tanh(x)`. Invalid domains, overflow, and any nonfinite result fail when written. Hardware `sfu_lanes` affects service time, not the functional formula.
- **`WAIT {"wg":string,"events":[string,...]}`**: the target must be active. Every event name must have been produced earlier in the expanded instruction stream, and names in the list must be unique. The target waits for those event completions. An empty events list is allowed and adds no event dependency.
- **`BARRIER {"wgs":[string,...],"events":[string,...]}`**: `wgs` is a nonempty, duplicate-free list of active groups; `events` follows the same rules as `WAIT`. Listed groups meet at the same barrier and wait for preceding work in its scope; unlisted groups are not participants. Use this to establish cross-group ordering when needed.
- **`STEP.COMMIT {"step":int}`**: steps start at 0 and must be consecutive. It globally waits for previous work and outstanding transfers, checks every required hidden/KV output for that step, and releases the next external input. P1 needs commit 0 after both 64-position prompts and before its first new input; the final new position is checked at case end. D1 needs commits 0–7, one after each decode output. A commit may occur after all workgroups end.

The legacy `WAIT {"events":[...]}` or `BARRIER {"events":[...]}` form is accepted only when exactly one workgroup is active. Explicit `wg`/`wgs` is recommended. A waited event must already be defined in source order. A workgroup preserves its own program order. Source-text interleaving alone does not order overlapping HBM reads/writes across workgroups; use a matching `WAIT`, a `BARRIER` covering the groups, or a cross-step `STEP.COMMIT`.

`WG.BEGIN` can create more logical active groups than the SM can run simultaneously; the timing engine queues them. At most four are physically resident per SM, and their declared SH quotas must fit. `WG.END` is required before the name can be reused. `STEP.COMMIT` also works when no group remains active.

## Vector kinds and operand counts

| `VEC kind` | Sources | Result |
| --- | ---: | --- |
| `add`, `sub`, `mul`, `div` | 2 | Corresponding FP32 arithmetic |
| `fma` | 3 | `a*b+c`, rounded to FP32 at the instruction boundary |
| `gt` | 2 | 1.0 where `a>b`, otherwise 0.0 |
| `select` | 3 | Choose second source where first is nonzero, otherwise third |
| `max` | 2 | Elementwise maximum |

A source RF view may have one element (broadcast) or the destination count. An immediate is `{"imm": 1.0}` and must round to finite FP32. `VEC`, `SFU`, `REDUCE`, and `MMA.ACC` may read different RF partitions of the **same** workgroup. `REDUCE` has a one-element destination. `MMA.ACC` sizes must be `m*k`, `k*n`, and `m*n`, respectively; the three matrices are row-major.

Vector source order is semantic: `sub` computes source 0 minus source 1, `div` divides source 0 by source 1, `fma` computes source 0 × source 1 + source 2, and `select` tests source 0 then picks source 1 or 2. `gt` uses a strict comparison. The executor rounds written values to FP32; `fma` forms its multiply-add in float64 before that FP32 write. Do not assume an arbitrary sequence of `VEC` instructions is bitwise equal to a fused `fma`. `div` by zero, `exp` overflow, and `rsqrt` of a negative value fail through the finite-write rule. Even if all values are finite, the scored outputs must also satisfy the reference tolerances in the [statement](/statement/#1-the-challenge).

`MMA.ACC` groups consecutive K terms by hardware `tc_k_parallel`. Each group uses a fixed neighboring binary reduction tree, then groups accumulate in ascending K order. A tail group is zero-filled. Ordinary `REDUCE sum` also uses a fixed binary tree. Each instruction rounds its output to FP32. Physical TC tail slots still cost time and energy; shorter views mask only out-of-range memory accesses.

## Static loops and expressions

A loop uses compiled integer bounds, for example:

```asm
FOR {"var":"i","start":0,"stop":3,"step":1}
VEC {"kind":"add","src":[{"space":"RF","offset":{"index":"i","scale":4,"offset":0},"count":4,"wg":"g","lane":0},{"imm":1.0}],"dst":{"space":"RF","offset":{"index":"i","scale":4,"offset":0},"count":4,"wg":"g","lane":0},"event":"add_{i}"}
END.FOR {}
```

An integer operand may use `{"var":"i"}`, the affine form `{"index":"i","scale":4,"offset":0}`, or `{"add":[...]}`, `{"mul":[...]}`, `{"min":[...]}`, `{"max":[...]}`, `{"mod":[a,b]}`, and `{"ceildiv":[a,b]}`. Expressions resolve from static loop indices. There is no runtime data-dependent branch or unbounded loop.

`FOR` has exactly `var`, `start`, `stop`, and `step`. The variable must be an identifier and cannot duplicate an enclosing loop variable. Bound values become nonnegative integers after substitution, `stop >= start`, and `step >= 1`. Iterations are `start, start + step, ...` strictly less than `stop`. An `END.FOR {}` closes one matching `FOR`. Every loop body is parsed even when its trip count is zero. Static integer expressions can be nested and can contain **negative integer constants inside arithmetic**, such as `{"mul":[-1,{"var":"k"}]}` for a tail calculation; the final operand must still satisfy its own range. `add`, `mul`, `min`, and `max` require at least two integer arguments; `mod` and `ceildiv` require exactly two with a positive divisor. Affine `scale` and `offset` must each be nonnegative integers. The same expressions can appear inside numeric fields of instructions and nested loop bounds. Strings containing braces interpolate a bound loop index, e.g. `"load_{i}"`; format specifications and conversions are unsupported. Give every expanded **event-bearing operation** a distinct event, including across loop iterations and reused workgroup names.

### Ordering and timing rules

The functional executor walks expanded instructions in source order to check values, but the performance model places workgroup instructions on separate streams. Program order is preserved **within** each workgroup. The text order of instructions from two different workgroups does not establish ordering for overlapping HBM reads and writes. The race checker rejects an unordered cross-group read/write or write/write overlap. To communicate through HBM, have the consumer wait on the producer's earlier event, place both groups in a barrier, or cross a `STEP.COMMIT` boundary. A `WAIT` event can be produced by another workgroup. `WAIT` does not automatically join all groups; `BARRIER` applies to exactly the listed groups. A commit is global.

Transfers have latency and may remain pending after issue. If a later operation uses data produced by a transfer, or overwrites a transfer's RF/SH source while a store is still reading it, the timed checker imposes the dependency; use explicit event waits when expressing cross-group intent. Local RF/SH and HBM resource service, DMA queues, TC/vector/SFU/reduction units, cache, NoC, and HBM all influence the score. The [public scheduler](/model/pipeline_events.py) and [service formulas](/model/service.py) define those timings. The functional ISA alone does not promise one cycle per instruction or unlimited parallelism.

### Before emitting a complete program

1. Read the [HBM ABI](/abi/) for each case's tensor base, shape, read-only status, output range, and step-release rule. P1 and D1 are distinct programs with distinct layouts. Hardware JSON is shared.
2. Allocate only legal RF lanes, RF offsets, and SH quotas for that hardware. Initialize accumulators and scratch before reading them. Use shaped views for noncontiguous matrix tiles; `count` is the logical element count, not the physical span.
3. Produce every required final hidden and new K/V value exactly once; commit P1 prompt and D1 steps at the required boundaries. End every workgroup and ensure event names stay unique after expansion.
4. Run the starter package's `python challenge.py check --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --seed 7 --report check.json` before running `estimate` or `grade`. See [the full development workflow](/assembly-guide/#7-generate-check-and-submit). The [complete P1](/examples/M1_P1.asm) and [D1](/examples/M2_D1.asm) listings provide executable patterns for every instruction used by the baseline.

## Small worked instruction sequence

For HBM values `[1,2,3,4,5,6,0,0]`, this program computes `1×4+2×5+3×6=32` into HBM element 7. This is an ISA illustration; full scored submissions must compute both complete Transformer cases.

```asm
WG.BEGIN {"wg":"g","sm":0,"shared_bytes":0}
LD {"src":{"space":"HBM","offset":0,"count":3,"wg":null,"lane":0},"dst":{"space":"RF","offset":0,"count":3,"wg":"g","lane":0},"event":"a"}
LD {"src":{"space":"HBM","offset":3,"count":3,"wg":null,"lane":0},"dst":{"space":"RF","offset":3,"count":3,"wg":"g","lane":0},"event":"b"}
LD {"src":{"space":"HBM","offset":6,"count":1,"wg":null,"lane":0},"dst":{"space":"RF","offset":6,"count":1,"wg":"g","lane":0},"event":"z"}
MMA.ACC {"a":{"space":"RF","offset":0,"count":3,"wg":"g","lane":0},"b":{"space":"RF","offset":3,"count":3,"wg":"g","lane":0},"acc":{"space":"RF","offset":6,"count":1,"wg":"g","lane":0},"m":1,"n":1,"k":3,"event":"mma"}
WAIT {"wg":"g","events":["mma"]}
ST {"src":{"space":"RF","offset":6,"count":1,"wg":"g","lane":0},"dst":{"space":"HBM","offset":7,"count":1,"wg":null,"lane":0},"event":"out"}
WG.END {"wg":"g"}
```

A write from RF or SH may continue reading its source after the `ST` is issued. Overwriting an overlapping source must wait until that read service finishes. All live groups must eventually end, every required step must commit, and every required output/KV word must be written. P1's required commit is after the prompt; D1 commits all eight decode steps. The complete baseline P1 and D1 assembly listings are available as [individual downloads](/examples/M1_P1.asm) and [the starter ZIP](/downloads/transformer-codesign-starter.zip).
