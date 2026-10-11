# Phase Two: five-case local self-test material map

The five static Phase Two programs are at the ZIP root under `programs/`.
`selftest/grade.json`, `selftest/contract/result.json`,
`selftest/workers/summary.json`, and each
`selftest/workers/<case>/evaluation/result.json` are the **unedited** outputs
of the supplied grader. Read `selftest/grade.json` for the local score,
contract/model/source identity, seed and the five actual cycle counts. The
five cases pass locally with **3441.4747199414096** points (seed 7), versus
**3883.1221920286375** points conditionally predicted by the model. This
is not a server-signed result; the organizer evaluates the same programs
with private seeds. `hardware.json` is extracted from the selected program
header and is identical across all five submitted programs.

## Material map

- `phase-two-model/README.md`, `MODEL.md` and `FORMULAS.html`: scope,
  mathematical formulation, limitations and human-readable equations.
- `phase-two-model/model/`, `tests/`, `requirements.txt` and `specs/`:
  solver/model, search code, tests, dependencies and declared experiment
  parameters. The finite-domain optimum is a *conditional prediction*;
  it does not prove the five programs are globally optimal.
- `phase-two-model/reports/`: recorded modeling revisions, baseline
  comparisons, this candidate's prediction-error analysis and examples of
  evidence-to-revision chains.
- `selection/`: original `screen.json`, `solution.json`, `model.json`,
  `preparation.json`, `spec.json`, `estimation.json`, `bound-screen.json`,
  `grade-command.log`, selected per-case generator configs and all eight
  original `program-choices/<case>/option-NNNN.jsonl` candidate sources.
  The five ZIP-root programs are the exact files snapshotted by the grader;
  their SHA256 values must match `selection/screen.json` and `grade.json`.
- `selection/host-environment.txt`: host system and language/runtime versions.
- `verify_archive.py`: checks the ZIP constraints, all five snapshot hashes,
  matching hardware/source/evaluation receipts and recomputed local score.
- `phase-two-starter/`: supplied Rust simulator source and Cargo lockfile,
  Python grading/packaging tools and published assignment/ABI/ISA contract;
  build caches are excluded.
- `phase-one/homework-submit-64214-20261009.zip`: separately evaluated
  strongest retained Phase One solution; its score is **not** a Phase Two
  result and cannot be compared to the five-case score.

Paths above are relative to `materials/0/phase-two-submission-materials-20261011/`
inside this ZIP. The grading program inputs and reports instead live directly
under `programs/` and `selftest/` in the ZIP.

Run `python3 materials/0/phase-two-submission-materials-20261011/verify_archive.py submission.zip`
after extraction to check the archived five-case result without rerunning the
simulator.

## Reproduce the self-test

After extracting the archive, from the archive root on a Linux host with
Rust 1.90+, Python 3.11+ and the locked Cargo dependencies:

```sh
cd materials/0/phase-two-submission-materials-20261011/phase-two-starter/source
cargo build --release --features compact --bin vnext-concurrent
cd ../../../../..
python3 materials/0/phase-two-submission-materials-20261011/phase-two-starter/tools/grade_v09.py programs reproduced-grade --binary materials/0/phase-two-submission-materials-20261011/phase-two-starter/source/target/release/vnext-concurrent --seed 7 --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 1200
```

For the model command and selected SHA256 values, see the top-level
`phase-two-model/README.md` and `selection/screen.json`. The original
model-first replay experiment refers to additional large intermediate
programs retained in the repository; they are not needed to evaluate the
submitted ZIP-root programs and are omitted from this archive. The
independent predicted score is not an official local or server score.

## Scope of this archive

This archive includes the executable programs, local grader receipts,
simulator/source and modeling record. It **does not** claim to contain a
complete multi-agent orchestration framework or native conversation exports.
Those additional process-evidence requirements remain outstanding for a
final, fully documented Phase Two assignment submission. No private
credentials or unreviewed raw session logs are embedded.
