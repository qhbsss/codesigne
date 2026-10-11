# Phase Two: Assignment Statement

> **Phase Two opens October 2, 2026 at 11:00 Beijing time.** Read the [Background and learning requirements](BACKGROUND.md) first. This page specifies the implemented machine and executable submission contract.

[TOC]

## What changes from Phase One, and why

| Change | Reason and consequence |
| --- | --- |
| MIP-driven multi-agent framework, with evidence and model revision | Make reusable modeling knowledge and search efficiency assessable alongside the final score. |
| Five larger scenarios, including batch 1/16 and long-history decode | Expose weight reuse, attention storage and latency tradeoffs; the full weights exceed the cache menu. |
| Paid RF capacity/ports, SH banking, cache contention, 250-cycle HBM readiness and finite outstanding requests | Distinguish access latency from service bandwidth and charge for hiding latency. |
| Explicit concurrent groups, DMA engines/queues, NoC and multicast | Make overlap and communication compete for physical resources. |
| 25 mm² equivalent area (100 AU); 34 W short-window and 26 W long-window limits | Charge capacity, compute and interconnect together. These are teaching coefficients, not a commercial GPU specification. |
| Rust simulator and static JSONL programs | Bound evaluation work and keep candidate execution independent of student host code. Old assembly and scores cannot be reused directly. |

The published bounds and formulas below define this teaching machine. A useful formulation may simplify them, but must state and validate those simplifications. Current trials show scenario-dependent winners; they do not prove that all memory levels are globally competitive or that search cannot find a simple winning family.

## 1. Task and five workloads

Design **one hardware configuration** and a separate program for each scenario. Minimize normalized latency while passing all numerical, capacity, area and power checks. Tiling, scheduling, layouts, packing, partial sums, fusion and mathematically equivalent algorithms are programmable. All computation and data movement must use paid primitives. No embedded answers, input-dependent program selection, or numerical-value-based skipping is allowed.

| Case | Layers / D / heads / FFN | Batch | Prompt or initial history | Decode steps | Main weights / initial KV |
| --- | --- | ---: | --- | ---: | --- |
| w-p | 2 / 1536 / 12 / 6144 | 1 | P=256 | — | 216 MiB / none |
| a-p | 2 / 512 / 8 / 2048 | 1 | P=2048 | — | 24 MiB / none |
| w-d1 | Same as w-p | 1 | T=4096 | 4 | 216 / 96 MiB |
| w-d16 | Same as w-p | 16 | T=256 | 4 | 216 / 96 MiB |
| w-d4l | Same as w-p | 4 | T=4096 | 4 | 216 / 384 MiB |

Main weights exclude small bias and normalization arrays. FP32 weight bytes are `4L(4D²+2DF)`; initial KV bytes are `8LBTD`. The two 96 MiB decode cases have different batch reuse and matrix shapes. A-P emphasizes long attention; materializing one layer's complete score tensor uses 128 MiB. Partial on-chip residency is a legitimate opportunity, not a failure of the workload design.

Every layer is pre-LayerNorm → QKV projections → scaled causal softmax attention → output projection and residual → LayerNorm → FFN with tanh GELU → residual. Both normalizations are affine, use variance divided by D and epsilon 1e-5; every projection has bias. GELU is `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`. No training, vocabulary projection, sampling, dropout, quantization or approximate sparse attention is included.

W scenarios share weights for the same seed/profile. Requests have independent input and KV. Hidden uses `[batch, token, D]`; KV uses `[time, batch, D]`. Attention cannot cross requests or heads. Each decode step sees `T+s+1` positions; its external input is released only after the preceding step's full output commit. Future steps cannot be batched early. Scenarios and seeds start cold.

All final hidden and every layer's new K/V are checked against an independent float64 reference rounded to FP32: `abs_error <= 1e-3 + 1e-3*abs(reference)`. NaN/Inf, uninitialized reads, missing outputs and out-of-bounds accesses fail. Finite fixtures are not a proof for arbitrary inputs.

## 2. All 23 hardware fields

Clock: **500 MHz** (2 ns/cycle). Area: **at most 100 AU**. Simulated HBM: **2 GiB**, including weights, KV, layout copies, scratch and outputs. Costs are teaching assumptions, not measured silicon or a commercial GPU prediction.

| Field | Values and conditions |
| --- | --- |
| `sms=N` | 2, 4, 8, 12, 16, 20, 24, 28, 32 |
| `rf_kib=R` | 16, 32, 64, 128, 256 per SM |
| `rf_tier=t` | 1/2/3: read 128/256/512, write 64/128/256 B/cycle |
| `sh_kib=S` | 0, 32, 64, 96, 128, 192, 256, 512, 1024 per SM |
| `sh_banks=b` | Zero if SH disabled; otherwise 1, 2, 4, 8, 16, 32, 64 with b ≤ S |
| `sh_tc_bw=B_t` | 0, 32, 64, 128, 256 B/cycle; nonzero requires SH and TC |
| `vector=V` | 8, 16, 32, 64, 128 lanes |
| `sfu=U` | Integer 1..V lanes |
| `p` | TC rows: 1, 2, 4, 8, 16, 32, 64; zero when disabled |
| `q` | TC columns: same menu; 16 ≤ p*q ≤ 512 when enabled; zero when disabled |
| `hbm_channels=h` | 1, 2, 4, 8 logical channels |
| `hbm_queue=Q` | 32, 64, 128, 256, 512 in-flight line transactions per channel |
| `sm_noc=B_s` | 16, 32, 64, 128, 256 B/cycle |
| `global_noc=B_n` | 64, 128, 256, 512 B/cycle |
| `tc_count=c` | 0, 1, 2, 3, 4, 6, 8 per SM; zero requires p=q=0 |
| `tc_k_parallel=k_p` | 1, 2, 4; use 1 when TC disabled |
| `reduction_units=r` | 0, 1, 2 per SM; zero routes reductions to Vector |
| `shared_ports=u` | 1 or 2 read ports and one write port per bank; disabled SH uses 1 |
| `dma_depth=z` | 0, 1, 2; each DMA engine queues 1+z descriptors; zero disallows Async |
| `dma_engines=d` | 1, 2, 4 per SM |
| `multicast` | false or true |
| `cache_mib=C` | Global 0, 1, 2, 4, 8, 16 MiB |
| `resident_groups=g` | 1, 2, 4, 8 workgroups per SM |

Each SM has one shared 32-entry unsent-request frontend and **128 pending slots covering all transfer stages**, each including a 64 B payload buffer and tags. Pending credit remains occupied until local acknowledgement. Cache, when enabled, has one shared 64 B/cycle interface and 256 MSHRs. Increasing capacity does not increase bandwidth. With an approximately 270–290-cycle end-to-end credit lifetime, 128 pending lines bound one SM to roughly 28–30 B/cycle of ideal reads. Eight SMs provide roughly 224–240 B/cycle before other bottlenecks, close to but not guaranteed to saturate eight HBM channels at 256 B/cycle. All combinations must pass area and local resource constraints.

## 3. Complete area and activity costs

**Area scale: 1 AU = 0.25 mm² equivalent logic-die area; the total budget is 25 mm² (100 AU).** All AU coefficients below convert by multiplying by 0.25. This is a declared teaching scale, not a measured process-node result. Logic, RF, SH, cache, interconnect and modeled HBM interfaces are included; external HBM dies and packaging are excluded from die area. Energy accounting retains the published HBM traffic charges. Unit conversion changes no legal configuration, timing, power or score. For example, 128 KiB tier-1 RF costs 0.576 mm²; 128 KiB SH capacity costs 0.128 mm² before bank/interface additions; an 8×16 K=1 TC costs 0.322 mm². Area-dependent base power is 0.025 W/AU = 0.1 W/mm².


All per-SM rows are multiplied by N. RF tier factor `f_t=1/1.6/2.6`. Disabled resources cost zero.

| Component | Area AU |
| --- | --- |
| Per-SM control / Vector / SFU | `0.45+0.018V+0.06U` |
| Per-SM TC | `c*(0.072+0.008*(pq+p+q)*k_p)` |
| Per-SM RF | `0.018R*f_t` |
| Per-SM SH | `0.004S+0.015b+(u-1)*(0.0015S+0.008b)` |
| Per-SM enabled SH-to-TC | `0.28+0.004B_t`, shared by all TC engines |
| Per-SM dedicated reduction | `r*(0.10+0.012V)` |
| Per-SM frontend + DMA | `0.514+d*(0.10+0.006*(4+2z))` |
| Per-SM pending buffers | `0.002*128=0.256` |
| Per-SM residency/token metadata | `0.08*(g-1)+0.002zg` |
| Per-SM NoC endpoint | `0.15*(B_s/64)^1.3` |
| Per-SM enabled multicast endpoint | 0.04 |
| Enabled global Cache | `3C+0.6+0.012*256+0.006*64` |
| HBM interfaces/slots | `0.65h+0.0015hQ` |
| Global NoC | `1+0.035N+0.8*(B_n/128)^1.3` |
| Enabled global multicast | 0.2 |
| Other fixed logic | 2 |

Cache's lower capacity price represents a slower, dense global memory macro with shared bandwidth. Private SH has different latency and concurrency. This is an explicit modeling assumption; the task does not force every architecture direction to score similarly. Cold traffic consumes both lookup and fill on the shared 64 B/cycle interface, giving an ideal 32 B/cycle ceiling. Little's law requires roughly 125 or more in-flight lines for a 250-plus-cycle miss path; 256 paid MSHRs leave return-path headroom. A fixed 32-entry table would unnecessarily limit cold traffic to roughly 7.5 B/cycle. Hot-hit service remains at most 64 B/cycle. There is no per-DMA Cache bypass in this version: an enabled Cache pays this cold-stream bottleneck against eight-channel HBM's ideal 256 B/cycle, so enough hits/reuse are needed to recover its cost. More MSHRs do not establish that Cache is a competitive choice; full workloads must test that tradeoff.

| Activity | Dynamic energy |
| --- | --- |
| RF reads/writes | `[0.30/0.42/0.60]*(1+0.1*log2(R/32))` pJ/B |
| SH reads/writes | `0.8*(1+0.05*log2(S/64))` pJ/B; 2R1W reads multiply by 1.15 |
| SH-to-TC | Extra 0.2 pJ/B |
| TC | 3 pJ/padded physical FMA; K-product tree adds 1.5 pJ/physical binary merge |
| Vector FMA / other | 4 / 1.5 pJ/effective element |
| exp, tanh / rsqrt | 12 / 8 pJ/effective element |
| Reduction | 1.5 pJ/actual binary merge, either engine |
| Cache lookup / fill / write invalidation | Each charges `(2.5+0.15*log2(C))` pJ/B for a full 64 B; miss pays lookup and fill |
| HBM | 150 pJ/B for full transactions; direction switch 200 pJ |
| NoC | Global trunk 1 pJ/B + destination endpoint 1 pJ/B, using each actual byte count |
| Instruction / Wait / fence issue | 8 pJ; fence charges once per SM |
| DMA startup / address generation | 20 pJ / 0.5 pJ per effective element |
| Frontend line / external transaction handling | 2 / 8 pJ per transaction |

Base power is `P0=0.025*area_AU+0.15*h` W. Maximum sliding-window average is **34 W over 100 cycles** and **26 W over 10,000 cycles**, including an idle tail. Concurrent activity adds on the same timeline; waiting does not spread a burst's energy. There is no automatic DVFS or free throttling. These windows are activity constraints, not a thermal or supply-network simulation.

The budget scale is explicit: one 32 B/cycle HBM channel at 500 MHz and 150 pJ/B consumes 2.4 W at peak. Eight channels use 19.2 W dynamically; a 100 AU chip adds 3.7 W base power, totaling 22.9 W before computation and on-chip traffic. The 26 W long window leaves 3.1 W for that activity, while the short window allows larger bursts. Dependencies and finite credits may prevent reaching peak HBM bandwidth. The thresholds are not tuned to reject a particular reference design.

## 4. Workgroups and bounded asynchronous DMA

A wave declares workgroups, their physical SM, RF/SH quotas, and commands. RF quotas use integer KiB (minimum 1 KiB); SH quotas must be multiples of 4 KiB. All groups on an SM must fit its separate RF/SH capacities and g slots. More slots do not copy physical storage. Local memory is private to a group; cross-group communication uses HBM and a wave barrier.

An SM issues at most one ready command per cycle with rotating group priority. A compute instruction blocks its group; other groups can use other TC, Vector, SFU or reduction engines. DMA can overlap computation. All engines contend for the same RF ports, SH banks, SH-to-TC interface, frontend and network. Service batches conservatively occupy their corresponding ports.

Commands are `run`, `async` and `wait`. Async accepts DMA only and requires z>0. Outstanding load destinations cannot be read or overwritten; outstanding store sources cannot be modified. Wait consumes the shared issue slot, one cycle and 8 pJ, and reclaims completed tokens. Missing/repeated tokens, conflicting accesses and unreclaimed tokens at group end fail.

Each DMA engine queues 1+z descriptors in FIFO order. A head descriptor completes before the next descriptor progresses; depth enables early enqueue, not a multiplication of transfer bandwidth. Each descriptor is at most 64 KiB and pays six startup cycles from enqueue, which may overlap the preceding descriptor. It progresses only when it reaches the head and startup has completed; depth can hide this successor startup without increasing service bandwidth. Across an SM, the frontend processes at most one 16-element address batch and sends at most two line transactions per cycle. Local staging and returns are bounded.

Unordered cross-group HBM read/write or write/write races fail even on the same SM; disjoint elements on one line can be legal. A wave drains computation, DMA, network and local returns, then pays a one-cycle global fence. RF/SH become invalid; Cache persists until scenario reset. No future decode input becomes available before a successful step commit.

## 5. Timing and memory hierarchy

| Resource | Service and dependency latency |
| --- | --- |
| RF | `ceil(bytes/bandwidth)` service + `2+floor(log2(max(R/32,1)))` cycles |
| SH | Bank service + `6+floor(log2(max(S/(4b),1)))` cycles; up to 14 cycles in this menu |
| SH ports | u distinct-word reads and one write/bank/cycle; same-word broadcast only within a 16-element service batch |
| SH-to-TC | SH completion → shared B_t service → 2 cycles |
| Cache | Shared 64 B/cycle service → `20+2*log2(C)` cycles |
| HBM | 250-cycle ready wait + 64 B/2 cycles/channel; read/write share service; turn costs 4 cycles |
| NoC | Both global and SM budgets; directions share capacity; 6-cycle propagation; multicast is trunk-then-endpoint store-and-forward |
| Vector / SFU | Each group reads, computes for 4/12 cycles, then writes; groups block |
| Reduction | Adjacent binary tree, at most V outputs per microstep; compute 4 cycles on Vector or 1 on dedicated reduction, with RF scratch traffic |

**One MMA occupies one TC.** Physical p×q output blocks execute serially within it; multiple workgroups can use multiple TC engines. Each block reads its accumulator once, then executes Kp microsteps of input-read → `1+log2(k_p)` compute cycles, then pays `p+q-2` drain and writes the accumulator once. Only one physical block's accumulator and one Kp input group can live internally. **These microsteps block; old pipelined K timing formulas do not apply.** Other groups can still overlap on other available resources.

RF/RF reads `4*activeK*(mm+nn)` bytes per Kp microstep. RF/SH uses actual B addresses and pays bank conflicts, interface service and dependency latency. Physical tails in p/q/K consume compute energy without out-of-bounds reads. Kp=1 preserves FP32 mul_add; Kp>1 rounds individual products to FP32, reduces with a fixed binary tree, then adds to the accumulator. All choices must pass the same numerical tolerance.

HBM lines are 64 B; channel is `(byte_address/256)%h`. DMA merges addresses only within each 16-element batch. Read requests send 8 B upstream and return 64 B. Writes send 72 B upstream (address + data) and return an 8 B completion notification. Credit remains until local acknowledgement. No DRAM row buffers, refresh, TLB or physical wire layout are modeled.

Cache uses 64 B lines, four ways and deterministic LRU. Same-line in-flight misses merge independently of multicast, with at most 32 destinations per transaction. Writes go through to HBM, do not allocate, and invalidate old cached copies. Releasing an allocation invalidates its address range at an idle boundary to prevent stale reuse; this is not a general software flush primitive.

With multicast off, each destination consumes 64 B on the return trunk. With it on, one transaction uses 64 B total on the trunk and 64 B at each destination endpoint. Each request still pays its own upstream address packet and local write. Requests arriving after return has started cannot retroactively join a broadcast. Cache miss merging saves HBM traffic; multicast separately saves trunk traffic.

## 6. Static programs and limits

Submit five static files: `w-p.jsonl`, `a-p.jsonl`, `w-d1.jsonl`, `w-d16.jsonl`, `w-d4l.jsonl`. Each UTF-8 JSON line is independent. The header contains contract, model, hardware, shape, batch and decode. All five use exactly the same hardware and fixed case shapes. Unknown fields are rejected.

Records are `input`, `alloc`, `release`, `wave` and `commit`. A wave contains groups with sm, rf_kib, sh_kib and commands. Inputs bind canonical weights, bias, LN parameters, historical KV and each step input exactly once. Bound inputs cannot be released. Layout changes use paid operations; allocations are 256 B aligned; scratch release is reverse-order after a completed wave.

Available primitives: HBM↔RF, HBM↔SH, SH↔RF, fill, vector arithmetic/SFU, sum/max reduction, RF/RF and RF/SH MMA. Views use FP32 element offsets, rows/columns and strides. MMA inputs cannot overlap its accumulator; vector exact in-place updates are allowed but partial aliases fail; reduction input/output/scratch are disjoint; DMA local destinations must be unique.

Commit supplies all final hidden and each layer's new K/V as contiguous tensor/base/length views. Validation is immediate; only a correct commit unlocks the next input. Prefill commits once and decode four times. Extra records after the final commit fail. The evaluator runs static primitives, never student host code.

| Limit | Bound |
| --- | --- |
| Simulated HBM / cumulative allocation | 2 GiB / 64 GiB |
| Line / program file / records | 16 MiB / 128 MiB / 200,000 source and expanded records |
| Total primitives / host-work units | 50 million / 200 billion; MMA counts mnk, other operations logical elements |
| Wave primitives / all commands | 1,048,576 / 2,097,152 |
| Groups per wave | At most N*g and 256; each SM has separate quota checks |
| DMA/local copy | 16,384 elements (64 KiB) |
| Fill/vector/reduce | 8,192 elements |
| MMA software tile | M,N ≤64, K ≤256; actual local capacity checked separately |
| Scenario / wave cycles | 8 billion / 2 billion, plus host event-work protection |

**Compact program controls.** Model `phase-two-compact-v2`, contract `phase-two-static-v1`, adds static `repeat`, `template`, `call` and checked integer address expressions. All expanded instructions remain paid. Loops within a wave retain local state; separate waves still invalidate RF/SH. See the [full syntax](ISA.md#compact-control-syntax). Loop count ≤1,000,000; control stack/expression depth ≤16; one expression ≤512 nodes; ≤4096 templates, ≤64 parameters each, ≤8 MiB total retained template text; ≤102 million control expansion steps. Multi-group HBM race detection retains at most 262,144 intervals after same-group contiguous/repeated range coalescing. Single-group waves need no cross-group table. These engineering limits are separate from hardware capacity. DMA and SH/RF copies require identical source/destination rows and columns; destination col_stride=1 and row_stride≥cols.

Compact representation reduced five complete candidate programs from about 584 MiB to 5.76 MiB. A 2048-position online-softmax kernel passed all 131,072 output comparisons after compact import. An independent trial found and reproduced an interval-count false rejection; v2 fixes the metadata representation while preserving paid operations and race checks. The five improved candidate programs retain exactly the same outputs, cycles, traffic and energy after this fix. Earlier long-wave experiments are historical evidence, not the current submission contract.

Research input admission also limits essential weights+KV to 1.5 GiB and projection work to 40 GFMA. These do not replace static-program or OS limits. The native generator exposes tile, packing, split-K, head grouping, persistent keys and SH reuse policies, plus `parallel_groups` and `async_prefetch`; students may generate other legal static algorithms.

| Native software field | Allowed values |
| --- | --- |
| `policy.m`, `policy.n`, `policy.k` | 1..64, 1..64, 1..256 |
| `policy.vector_gemv`, `policy.pack_transpose` | Boolean |
| `policy.split_k` | 1, 2, 4, 8 |
| `policy.head_group` | 1..32 |
| `policy.persistent_keys`, `policy.sh_direct` | Boolean; actual hardware/capacity still checked |
| `policy.sh_rows` | 0..8; zero disables this reuse strategy |
| `parallel_groups` | 1..resident_groups |
| `async_prefetch` | Boolean; Async commands require dma_depth > 0 |
| `fixture_profile` | legacy or attention_stress (default) |
| `reference_threads` | 1..8, default 1; host reference only |

This is a reference-generator menu, not the full static algorithm space. Packing, persistent layouts and partial sums pay their actual operations and storage. Enumerating these policy fields does not enumerate all valid programs.

## 7. Local grading and assessment

The implemented score is `1000*exp(sum(w_i*ln(reference_cycles_i/cycles_i)))`: each Prefill has weight 1/4, each Decode 1/6. All numerical, power and resource gates must pass. Reference cycles are frozen at 187689274/374856358/27068433/42358913/70439292 for w-p/a-p/w-d1/w-d16/w-d4l. Scores have no upper cap.

From the simulator source directory containing Cargo.toml, build with `cargo build --release --features compact --bin vnext-concurrent`. The CLI supports `check`, `export`, `evaluate` and `grade`, taking a config/program path, seed and a new output directory. A bounded parallel wrapper is available as `tools/grade_v09.py`: it snapshots all five programs, checks their shared hardware and fixed shapes, launches CPU/memory-budgeted workers, and verifies program/source hashes before aggregating results. Default grading uses two workers, 4 GiB address space per Linux worker, a 300 s adjustable wall limit and a 64 MiB log limit. Local results are reproducible reports, not signed server receipts; organizers rerun the same program hashes with private seeds.

The five-workload baseline has passed complete numerical execution. Native seed 7 and independent static seed 29 produce exactly identical simulated reports; the frozen static reference scores 1000. Four-worker static grading took 144.761 s including snapshots, with sampled aggregate RSS of 3.097 GiB. Ten candidate families were tested; nine pass all five cases, while cache-reuse retains two 300 s timeouts and receives no score. There were 55 candidate attempts, including six RF-quota rejections and two timeouts, plus five native and five static reference checks. Final complete-family scores are: reference 1000; dual-tc 829.056; rf-only 1286.915; sh-bandwidth 758.329; cache-tiled 955.432; k4-compact 1203.112; sm12-skinny-k2 1184.195; k4-rf 1948.876; k4-sh-staged 1548.994.

k4-rf wins the tested Prefill and batch-16 Decode cases, while sm12-skinny-k2 wins batch-1 and long-KV Decode (18.979/51.153 million cycles versus 19.496/56.694 million for k4-rf). This establishes scenario tradeoffs among tested programs. The best aggregate remains a no-SH design; global competitiveness of SH/Cache has not been established. These trials neither prove optimality nor rule out eventual convergence to a no-SH family.

Eight complex synthetic kernels (two near-limit MMA, six large-stride DMA) completed in 9.862 s with four workers and 5.034 s with eight; sampled aggregate RSS was 624.52 and 959.65 MiB. All 16 runs passed numerical and power checks, with exactly matching simulated reports across parallelism settings. These are kernel stress results, not full Transformer timings; summed RSS double-counts shared pages and may miss brief peaks.

Report raw case cycles, decode step latency, throughput, energy, both power windows, HBM traffic, NoC trunk/endpoint traffic, peak storage and evaluator wall time/RSS. Batch latency divided by B is not individual request response latency. Simulated HBM capacity is not host process memory.

No tested area-legal program has yet exceeded the 26/34 W thresholds. Passing candidates establish executable accounting and checks; they do not establish that the power limits exclude a nonempty part of the area-legal space. This remains an explicit validation gap.

The learning objective is for an agent to identify useful variables, model capacity/communication/concurrency, predict changes and test counterexamples. A meaningful MIP formulation is required in Phase Two. Decomposition, network-flow subproblems, analytical bounds and hybrid search may support that formulation. Solver usage alone is not evidence of a useful model; no guarantee is made that modeling always beats direct search or that multiple architecture families have equal optimal scores.

The compact-v2 repair passed 115 Rust all-feature tests, 106 legacy concurrent tests, 464 project Python tests and all-target Clippy. Independent five-case seed-73 re-evaluation retained score 1965.1728 and all simulated metrics; finite trials do not prove optimality.


## 8. Submission materials and process evidence

Use the [separate Phase Two submission page](https://linux-slai.tail6d76d1.ts.net:8443/phase-two/submit/). Phase One continues to use its existing form and API. Phase Two opens October 2 at 11:00 Beijing time. The deadline remains October 12 at 12:00 Beijing time. Use the packaging tool on the submit page to assemble the required machine-readable paths; report and native trace organization stays flexible.

**Executable result.** Supply the five static JSONL programs listed above, the generated hardware configuration, and the unedited local `grade.json` plus the corresponding evaluation reports. The hardware embedded in all five headers must match. These machine-readable contracts are strict. Include source, configuration, dependencies and commands needed to regenerate and evaluate the result. A local report is evidence, not a verified server score.

**Required local self-test results.** As in Phase One, submit your own self-test results for the exact final programs. Include the evaluator-generated `grade.json`, `contract/result.json`, `workers/summary.json`, and each of the five `workers/<case>/evaluation/result.json` reports (`w-p`, `a-p`, `w-d1`, `w-d16`, `w-d4l`). These paths identify outputs of the supplied parallel grading tool; you may organize your archive differently if the material map locates them. Retain the machine-readable originals, not just screenshots or a manually transcribed score. Record the evaluation command, seed, simulator model/contract/source hash, program hashes and host environment. The reports must cover numerical correctness, resource/power feasibility, case cycles and the aggregate score. Preserve failure reports and logs for incomplete runs and clearly identify missing cases; never present a partial run as a passing five-case result. Re-run the self-test after changing a final program. Local results support reproduction; the organizer re-evaluates the submitted programs with private seeds for the final score.

**Separate size allowances.** Upload one ZIP of at most **256 MiB**, with at most **1 GiB expanded contents and 10,000 entries**. Programs, self-tests, framework/source, report and native traces share that material allowance. Each executable independently has a 128 MiB limit; supplementary material does not consume its program quota. Remove dependency/build caches and redundant intermediates; keep relevant original process records. Symbolic links, unsafe paths, duplicate ZIP entries and encrypted archives are rejected.

**Framework and formulation.** Include the runnable multi-agent framework, MIP definition, solver settings/status/gap, Phase One evidence, pruning rules and an analysis report. Preserve your strongest previous solutions; compare legal regenerated baselines under the same revised simulator and comparable budgets. Explain the hardware–software coupling, major discrepancies, revisions and your own contributions. Include one real, traceable evidence-to-revision chain. A fixed number of failed attempts is not required.

**Native process records.** Export relevant available sessions from Codex, Claude Code, DeepSeek harness, ZCode or your chosen tools, including relevant failed paths, restarted sessions and subagent records. JSON, JSONL, HTML, Markdown, text and other viewable native exports are accepted. Multiple sessions may remain separate. Add a short material map explaining their roles and order; there is no mandatory trace schema, manifest, directory layout or report format. Do not transcribe every tool call or fabricate missing logs. Explain lost, compacted or unavailable portions and their impact. Inaccessible internal reasoning is not requested.

**Evidence and metrics.** Locate key claims by existing file paths, experiment IDs, timestamps or commits. Link the final result to the actual simulator/configuration/program versions. Count failed runs, timeouts and retries in search cost. Distinguish wall time from accumulated parallel worker time. Use existing token counters; label missing/partial/estimated figures with scope and method, never substitute zero for unavailable data. Do not rerun solely to recover counters. Mark retrospective explanations and unimplemented proposals explicitly.

**Privacy and responsibility.** Redact API keys, SSH private keys, passwords, tokens, cookies, authentication links and unrelated private data before uploading. Use placeholders such as `[REDACTED_API_KEY]` and briefly state the redaction scope; proper redaction is not missing evidence. Do not upload an entire home directory, environment or agent configuration. If a credential leaks, revoke or rotate it and contact the instructor. Automated work and AI-assisted writing are allowed; verify factual claims and be prepared to explain the key decisions. Manual prompt counts, agent counts and log length are not assessment targets. Process materials receive human review, not automatic grading based on trace fields.

## 9. Downloads and local execution

| Download | Contents and compatibility |
| --- | --- |
| [Phase Two complete starter](https://linux-slai.tail6d76d1.ts.net:8443/downloads/phase-two-starter.zip) | Complete Rust source, Cargo.lock, Python 3.11+ scheduling/grading tools, ten reference families, Background, Statement, ISA, ABI, walkthrough and a complete small executable example. Rust 1.90+ is needed to compile. |
| [macOS ARM64 executable](https://linux-slai.tail6d76d1.ts.net:8443/downloads/phase-two-macos-arm64.zip) | Convenience binary tested on this Apple Silicon macOS 27.0 host; no promise for older macOS releases. Not a Linux, Windows or Intel Mac executable. Includes checksum and source identity. |
| [Linux ARM64 executable](https://linux-slai.tail6d76d1.ts.net:8443/downloads/phase-two-linux-arm64.zip) | Tested on DGX Spark Ubuntu ARM64 with the complete five-case evaluation. Not an x86 or Windows executable; source builds are available. |
| [Phase One Python starter](https://linux-slai.tail6d76d1.ts.net:8443/downloads/transformer-codesign-starter.zip) | Existing Python/NumPy grader and old programs. Implements Phase One rules only. |

Phase Two has a **Rust simulation core and Python orchestration tools**. There is no independent Python implementation of the Phase Two timing model. Rust source is provided so you can compile for your own machine. Linux ARM64 builds have passed the complete five-case private-seed evaluation on DGX Spark; a convenience binary is provided. Windows and x86 builds have not been validated. First compilation needs access to the locked Cargo dependencies. On Linux the runner supports address-space limits; macOS does not enforce that particular hard limit.

After extracting the source ZIP, run these commands from its root. Use new output directories for each attempt:

```sh
mkdir -p runs
cargo build --release --manifest-path source/Cargo.toml --features compact --bin vnext-concurrent
source/target/release/vnext-concurrent check configs/reference/w-d1.json 7 runs/check-d1
```

Generate the five static programs and grade them:

```sh
mkdir -p programs runs
for case_name in w-p a-p w-d1 w-d16 w-d4l; do
  source/target/release/vnext-concurrent export "configs/reference/$case_name.json" 7 "runs/export-$case_name"
  python3 tools/compact_program.py "runs/export-$case_name/program.jsonl" "programs/$case_name.jsonl"
done
python3 tools/grade_v09.py programs runs/grade --binary source/target/release/vnext-concurrent --seed 7 --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 600
```

The output is `runs/grade/grade.json`, with per-case worker reports beneath that directory. Preserve failed outputs too. Increase parallelism only within your CPU and memory budget. The package includes compact reference programs, the compactor, and `tools/package_phase_two.py`. The release model is `phase-two-compact-v2`, contract `phase-two-static-v1`; the executable retains the name `vnext-concurrent`. Run `python3 tools/package_phase_two.py runs/grade submission.zip --materials YOUR_MATERIALS` to assemble programs, required self-tests and supplementary materials.
