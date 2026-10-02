"""Package a fresh official Linux grade without rewriting any report fields."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "official-linux"
OLD = HERE.parent / "homework-complete-28885-20261002"
DEST = HERE.parent / "homework-submit-linux-20261002"
ZIP = DEST.with_suffix(".zip")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def digest_json(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    assert not DEST.exists() and not ZIP.exists(), "Do not overwrite an existing package"
    report_path = OFFICIAL / "local-grade.json"
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    assert report["mode"] == "grade" and report["eligible"] is True
    assert report["runtime_environment"]["platform"].startswith("Linux")
    assert report["seed_sha256"] == [digest(b"challenge-seed-v1:7")]
    assert report["gate_diagnostics"]["area_gate_passed"]
    with zipfile.ZipFile(HERE / "official-starter.zip") as z:
        published = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    source_names = sorted(n for n in published if n.startswith("codesign/challenge/")
                          and n.count("/") == 2 and n.endswith(".py")
                          and Path(n).name not in {"cli.py", "official.py", "release.py", "report_view.py"})
    source_names += sorted(n for n in published if n.startswith("codesign/challenge/cost_") and n.endswith(".json"))
    source_names += sorted(n for n in published if n.startswith("codesign/challenge/workload_") and n.endswith(".json"))
    expected_sources = {n: digest(published[n]) for n in source_names}
    p = report["provenance"]
    assert p["source_sha256"] == expected_sources, "Not the published Linux release"
    assert p["simulator_sha256"] == digest_json(expected_sources)
    for name in source_names:
        assert (OFFICIAL / name).read_bytes() == published[name]
    for name in ("cli.py", "official.py", "release.py"):
        assert report["trust_policy_sha256"][name] == digest(published["codesign/challenge/" + name])
    assert report["baseline_manifest_sha256"] == digest(published["baseline_manifest.json"])
    assert (OFFICIAL / "baseline_manifest.json").read_bytes() == published["baseline_manifest.json"]
    assert p["hardware_sha256"] == digest_json(json.loads((OFFICIAL / "hardware.json").read_text()))
    for case in ("M1_P1", "M2_D1"):
        path = OFFICIAL / "programs" / (case + ".asm")
        assert path.read_bytes() == (OLD / "programs" / path.name).read_bytes()
        assert digest(path.read_text(encoding="utf-8").encode()) == p["program_sha256"][case]
        gates = report["gate_diagnostics"]["scenarios"][case]
        assert gates["functional_passed"] and gates["power_gate_passed"] and gates["latency_gate_passed"]
    assert (OFFICIAL / "hardware.json").read_bytes() == (OLD / "hardware.json").read_bytes()
    DEST.mkdir()
    shutil.copytree(OFFICIAL / "codesign", DEST / "codesign", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("challenge.py", "baseline_manifest.json", "README.md", "ASSEMBLY_GUIDE.md", "ABI.md", "ISA.md", "BACKGROUND.md", "hardware.json", "local-grade.json"):
        shutil.copy2(OFFICIAL / name, DEST / name)
    shutil.copytree(OFFICIAL / "programs", DEST / "programs")
    (DEST / "project").mkdir()
    for name in ("__init__.py", "compiler.py", "schedule.py", "functional_check.py", "rough_model.py", "exact_case.py", "iteration-log.md"):
        shutil.copy2(OLD / "project" / name, DEST / "project" / name)
    shutil.copytree(OLD / "project/reports", DEST / "project/reports")
    for name in ("requirements.txt", "homework-announcement.md"):
        shutil.copy2(OLD / name, DEST / name)
    for name in ("verify_submission.py", "COMPATIBILITY.md"):
        shutil.copy2(HERE / name, DEST / "project" / name)
    shutil.copy2(HERE / "SUBMISSION.md", DEST / "SUBMISSION.md")
    (DEST / "agent-trace").mkdir()
    shutil.copy2(OLD / "agent-trace/README.md", DEST / "agent-trace/README.md")
    traces = []
    sessions = Path(r"C:\Users\qiuhb\.codex\sessions\2026\10\01")
    for old in sorted((OLD / "agent-trace").glob("*.jsonl")):
        source = sessions / old.name
        target = DEST / "agent-trace" / old.name
        shutil.copy2(source, target)
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        traces.append({"file": target.name, "records": len(rows), "sha256": digest(target.read_bytes())})
    stamp = datetime.now(timezone.utc).isoformat()
    dump(DEST / "agent-trace/trace-index.json", {"indexed_at_utc": stamp, "sessions": traces})
    dump(DEST / "project/linux-grade-summary.json", {
        "grade_report_sha256": digest(report_raw), "local_score": report["experimental_score"],
        "official_starter_sha256": digest((HERE / "official-starter.zip").read_bytes()),
        "official_starter_url": "https://linux-slai.tail6d76d1.ts.net:8443/downloads/transformer-codesign-starter.zip",
        "official_linux_source_match": True, "server_receipt_verified": False,
        "gates": report["gate_diagnostics"], "runtime_environment": report["runtime_environment"],
        "runtime_observation": report["runtime_observation"], "packaged_at_utc": stamp})
    manifest = {p.relative_to(DEST).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
                for p in sorted(DEST.rglob("*")) if p.is_file()}
    dump(DEST / "project/file-manifest.json", {"algorithm": "sha256", "files": manifest,
                                               "excludes": ["project/file-manifest.json"]})
    with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(DEST).as_posix())
    assert ZIP.stat().st_size < 25 * 1024**2
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert z.read("local-grade.json") == report_raw
        assert set(z.namelist()) == set(manifest) | {"project/file-manifest.json"}
        for name, info in manifest.items():
            assert digest(z.read(name)) == info["sha256"]
    print(json.dumps({"zip": str(ZIP), "mib": ZIP.stat().st_size / 1024**2,
                      "sha256": digest(ZIP.read_bytes()), "local_score": report["experimental_score"],
                      "server_verified": False}, indent=2))


if __name__ == "__main__":
    main()
