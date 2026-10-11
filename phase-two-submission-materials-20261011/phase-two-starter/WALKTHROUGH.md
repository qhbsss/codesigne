# Phase Two: Starter and program walkthrough

[Download the complete starter](https://linux-slai.tail6d76d1.ts.net:8443/downloads/phase-two-starter.zip). It contains `BACKGROUND.md`, `README.md` (statement), `ISA.md`, `ABI.md`, `WALKTHROUGH.md`, Rust source with Cargo.lock, Python runners, ten configuration families and a complete small static program. All tools run locally; no portal access is required.

## Build and check a complete small example

Requires Rust 1.90+; Python runners require Python 3.11+ and standard library only. From the extracted package root:

```sh
mkdir -p runs
cargo build --release --manifest-path source/Cargo.toml --features compact --bin vnext-concurrent
source/target/release/vnext-concurrent evaluate examples/small-prefill.jsonl 7 runs/small-check
```

This executes a complete small Transformer with independent numerical verification. It is an educational example and does not have the official workload shape, so it cannot receive an official five-case score. Its header and each record are genuine evaluator inputs. `examples/small-config.json` reproduces it with `export`.

## Reference configurations

All five official reference configs share one hardware object:

- [W-P config](examples/w-p.json): weight-heavy prefill.
- [A-P config](examples/a-p.json): long attention prefill.
- [W-D1 config](examples/w-d1.json): batch-one decode.
- [W-D16 config](examples/w-d16.json): batch-sixteen decode.
- [W-D4L config](examples/w-d4l.json): long-history batch-four decode.
- [Hardware only](examples/hardware.json).
- [Complete small program](examples/small-prefill.jsonl) and [its generator config](examples/small-config.json).

Five complete compact reference programs are included in `programs/reference/` (about 5.76 MiB total). The native exporter creates larger intermediate files; compact them before submission. Use a fresh `runs` output directory for every attempt.

```sh
mkdir -p programs runs
for case_name in w-p a-p w-d1 w-d16 w-d4l; do
  source/target/release/vnext-concurrent export "configs/reference/$case_name.json" 7 "runs/export-$case_name"
  python3 tools/compact_program.py "runs/export-$case_name/program.jsonl" "programs/$case_name.jsonl"
done
python3 tools/grade_v09.py programs runs/grade --binary source/target/release/vnext-concurrent --seed 7 --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 600
```

Read `runs/grade/grade.json` and the per-case reports. Failed correctness/resource checks or timeouts do not earn a score. This Python script schedules the Rust evaluator; it is not an independent Python timing implementation. See the statement's downloads section for binary platform support.

## Follow the generated program

1. **Header:** pins hardware, workload shape and contract. Export does not embed seed-dependent input values.
2. **Input bindings:** allocate the canonical weights and current inputs; tensor IDs follow allocation order. Track these IDs in your generator.
3. **Scratch allocation:** reserve intermediate HBM tensors; choose layouts and sizes under the ABI limits.
4. **Waves:** reserve each group's RF/SH quota, load tiles, initialize accumulators, execute MMA/vector/reduction operations and store results. RF/SH is local to the wave; HBM carries persistent state.
5. **Asynchronous operations:** load into a separate buffer, compute on a ready buffer, then wait before consumption/reuse. Capacity and bandwidth still compete on the physical SM.
6. **Commit:** supply final hidden plus each layer's new K/V. A decode program repeats the verified step boundary four times.

The JSONL ISA has no host callbacks or data-dependent program selection. A solver may produce hardware, tiling, placement and scheduling choices; your generator must turn those into explicit paid instructions. The reference generator's `policy` is a starting point, not the full algorithm space.

## Reproduction and assessment

Keep the exact source/config/program hashes, solver status and local reports with each attempt. The ten provided families are reference experiments, not guaranteed good or optimal solutions; `cache-reuse` includes known timeout cases. Reports remain unsigned local results. Supply native process records and the modeling report according to the [submission requirements](README.md#8-submission-materials-and-process-evidence).


## Assemble the submission

Use `python3 tools/package_phase_two.py runs/grade submission.zip --materials YOUR_SOURCE --materials YOUR_REPORT --materials YOUR_NATIVE_TRACES`. The tool includes the exact graded program snapshots and required self-test outputs. Upload one ZIP≤256 MiB; expanded contents≤1 GiB and≤10000 entries. Keep relevant native logs; omit dependency caches and redundant generated intermediates. See the [submission page](https://linux-slai.tail6d76d1.ts.net:8443/phase-two/submit/) for the required machine-readable paths and opening time.
