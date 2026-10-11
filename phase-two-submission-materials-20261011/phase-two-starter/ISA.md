# Phase Two: Static program and instruction reference

Programs are UTF-8 JSONL: one header, then one complete JSON object per line. Blank/comment lines are not part of the format. Unknown fields are rejected. Field names and tagged enum strings are case-sensitive. This format replaces Phase One assembly.

[TOC]

## Header and records

The header fields are `contract`, `model`, `hardware`, `shape`, `batch`, `decode`. Use contract `phase-two-static-v1` and model `phase-two-compact-v2`. `shape` contains `layers,d,heads,f,prefill,history,steps`; official values and the full hardware object are available in the [five reference configs](WALKTHROUGH.md#reference-configurations). All five official programs must embed identical hardware. Exporting a supplied config generates an exact header.

| `record` | Additional fields | Meaning |
| --- | --- | --- |
| `input` | `source,len` | Bind a canonical evaluator input |
| `alloc` | `len` | Allocate uninitialized HBM scratch |
| `release` | `tensor` | Pop last live scratch allocation |
| `wave` | `groups` | Run explicit concurrent groups; global completion boundary |
| `commit` | `outputs` | Validate final hidden and each layer's new K/V for this step |

Input names, layouts, tensor IDs, views and lifetimes are specified in the [Memory ABI](ABI.md).

## Groups and commands

A group has `sm,rf_kib,sh_kib,commands`; `sm` is zero-based. Reservations are explicit. Multiple groups may target one SM subject to summed capacity and resident-group limits. Groups compete for physical resources. All groups finish before the next record runs.

| `command` | Fields | Meaning |
| --- | --- | --- |
| `run` | `instruction` | Blocking instruction in this group |
| `async` | `token,instruction` | Asynchronous HBM load/store, including shared variants |
| `wait` | `tokens` | Wait for the named outstanding transfers |

Tokens are unsigned 32-bit integers, unique within a group's command list. Each async token needs exactly one explicit wait; no unknown/repeated tokens or empty waits. Async requires `dma_depth>0`. Pending transfers cannot race with later accesses to overlapping locations; wait before reusing the buffer. The wave boundary drains work, but does not replace the explicit wait requirement.

```json
{"record":"wave","groups":[{"sm":0,"rf_kib":1,"sh_kib":0,"commands":[{"command":"run","instruction":{"op":"fill","dst":0,"len":16,"value":0.0}}]}]}
```

This is a valid instruction example, not a full Transformer program.

## Instruction fields and semantics

All instruction objects use an `op` tag. All fields below are required, including `b` for unary vector instructions. RF/SH bases and lengths count FP32 elements. HBM `global` and RF/SH `local` use the view object documented in the ABI.

| `op` | Additional fields | Operation |
| --- | --- | --- |
| `load` | `tensor,global,local` | HBM → RF |
| `store` | `tensor,global,local` | RF → HBM |
| `load_shared` | `tensor,global,local` | HBM → SH |
| `store_shared` | `tensor,global,local` | SH → HBM |
| `shared_read` | `shared,local` | SH → RF |
| `shared_write` | `shared,local` | RF → SH |
| `fill` | `dst,len,value` | RF contiguous constant fill |
| `vector` | `kind,dst,len,a,b` | Elementwise RF/immediate computation |
| `reduce` | `scratch,dst,src,len,max` | RF sum (`max=false`) or maximum (`true`) into one RF element, using RF scratch |
| `mma` | `a,b,c,m,n,k` | RF row-major A[m,k] × B[k,n], accumulate into C[m,n] |
| `mma_shared` | `a,b,c,m,n,k` | A and C in RF; B is a SH view with shape [k,n] |

Vector arguments are `{"kind":"reg","base":0,"stride":1}` or `{"kind":"imm","value":1.0}`. A zero register stride broadcasts. Kinds: `add`, `sub`, `mul`, `div`, `fma`, `exp`, `tanh`, `rsqrt`, `square`. Binary operations use a and b; `fma` computes `a*b+old_dst` with fused FP32 arithmetic. Unary operations use a. Destination is contiguous. Exact in-place vector updates are allowed; partial aliases are rejected. MMA inputs may not overlap its accumulator. Reduction source/output/scratch must be disjoint; reserve `len` scratch elements. Transfer source and destination must have identical `rows` and `cols`; equal element counts alone do not allow reshape. For HBM↔RF, HBM↔SH and SH↔RF transfers, the destination must have `col_stride = 1` and `row_stride >= cols`, including single-row transfers. Source strides may represent broadcasts or strided reads. General scatter destinations are unsupported; use explicit paid transfers into legal contiguous rows. Reads of uninitialized data cannot produce a passing output.

MMA software limits are m,n ≤64, k ≤256. Physical p/q/K parallelism, tail cost and FP32 summation order are hardware choices; the [statement](README.md) defines their timing. A single MMA uses one TC; other groups are needed to use other engines concurrently.

## Resource limits and timing

Maximum transfer is 16,384 elements; fill/vector/reduce 8,192. Per wave: 1,048,576 primitives and 2,097,152 commands. Per program: 200,000 records, 50 million primitives, 200 billion host-work units, 128 MiB file and 16 MiB per line. At most 256 groups and `sms*resident_groups` groups per wave, with per-SM checks. These caps do not waive storage, numerical, power or cycle constraints.

Wave issue, RF ports, SH banks/interface, engines, DMA slots, HBM credits/cache/NoC and power are charged by the simulator. Async overlap has no zero-cost data movement exemption. For exact latency, area, energy and arbitration rules, use the [statement](README.md). Rust source `isa.rs`, `v09_schedule.rs` and `v09_submission.rs` provides the complete executable grammar; the package includes it.


## Compact control syntax

A top-level loop is `{"record":"repeat","var":"i","start":0,"count":3,"step":1,"body":[...]}`. Inside a group's commands use `"command":"repeat"` with the same fields and a command body. A command loop adds no wave boundary. Counts are integers 0..1,000,000, step is nonzero, body is nonempty and loop names cannot shadow an existing variable. Names have 1..32 ASCII letters, digits or underscores.

Integers may be literals, `{"var":"i"}` or binary expressions `{"add":[a,b]}`; operations: add, sub, mul, div, mod, min, max. Arithmetic is checked signed i64; division truncates toward zero. Overflow, division by zero, unbound variables and negative addresses fail. Expressions cannot inspect tensor values, seeds or host state. Expression depth≤16 and nodes≤512.

Define a top-level template with `{"record":"template","name":"t","kind":"record","params":["n"],"body":[...]}`. Invoke it with `{"record":"call","name":"t","args":[1024]}`. Command templates use kind `command` and `command:"call"`. Arguments are explicit; outer variables are not implicitly captured. Combined control stack depth≤16, at most4096 templates, 64 parameters each and 8MiB retained template text. Duplicate JSON keys and extra source records after final commit fail. Async tokens must remain unique across the fully expanded group, including iterations.

Total expansion work≤102 million. For multi-group waves, the HBM race checker retains at most262144 intervals after same-group continuous/repeated interval coalescing; single-group waves skip that table. These bounds do not remove instruction work. The evaluator materializes one expanded wave at a time; a compact source file does not imply cheap execution.
