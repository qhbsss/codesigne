"""Read-only checks for the frozen submission; does not rerun full grading."""
from pathlib import Path
import hashlib
import json

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.runner import provenance, seed_digest, digest_json
from project.schedule import generate_m1_p1, generate_m1_d1

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "5af2774ed4c947b5627f5a85b8c7f5e78f52797b0aa01df14833ac8943ab2312"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    report = json.loads((ROOT / "local-grade.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "local-grade.json") == REPORT_SHA256, "Original report changed"
    hw = Hardware.from_dict(json.loads((ROOT / "hardware.json").read_text()))
    programs = {case: (ROOT / "programs" / f"{case}.asm").read_text(encoding="utf-8")
                for case in ("M1_P1", "M2_D1")}
    actual = provenance(hw, programs)
    expected = report["provenance"]
    # Original Windows report uses backslashes in source names. Preserve it;
    # compare portable source keys while verifying the original aggregate hash.
    assert digest_json(expected["source_sha256"]) == expected["simulator_sha256"]
    assert {k.replace("\\", "/"): v for k, v in actual["source_sha256"].items()} == {
        k.replace("\\", "/"): v for k, v in expected["source_sha256"].items()}
    for key in ("hardware_sha256", "program_sha256", "cost_sha256", "workload_sha256"):
        assert actual[key] == expected[key], key
    assert sha(ROOT / "baseline_manifest.json") == report["baseline_manifest_sha256"]
    assert report["seed_sha256"] == [seed_digest(7)]
    assert report["eligible"] and report["gate_diagnostics"]["eligible"]
    assert report["experimental_score"] == 28885.37935367514
    result = {"kind": "submission-package-check-not-new-grade", "passed": True,
              "report_unchanged": True, "provenance_match": True,
              "seed": 7, "local_score": report["experimental_score"], "programs": {}}
    for case, generate in (("M1_P1", generate_m1_p1), ("M2_D1", generate_m1_d1)):
        text = programs[case]
        assert generate()[0] == text, f"Generator mismatch: {case}"
        depth = peak = commits = 0
        for line in text.splitlines():
            op = line.split(maxsplit=1)[0] if line.strip() else ""
            if op == "FOR":
                depth += 1
                peak = max(peak, depth)
            elif op == "END.FOR":
                depth -= 1
            commits += op == "STEP.COMMIT"
        count = sum(1 for _ in iter_parse(text))
        size = (ROOT / "programs" / f"{case}.asm").stat().st_size
        assert depth == 0 and size < 8 * 1024**2 and count < 10**7 and peak < 32
        assert report["cases"][case]["functional_passed"]
        if case == "M2_D1":
            assert commits == 8
        result["programs"][case] = {"generator_matches": True, "bytes": size,
            "expanded_instructions": count, "max_loop_depth": peak, "step_commits": commits}
    traces = []
    for path in sorted((ROOT / "agent-trace").glob("*.jsonl")):
        records = 0
        metadata = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                records += 1
                if metadata is None and record.get("type") == "session_meta":
                    metadata = record["payload"]
        assert metadata and metadata["cwd"].lower() == "d:\\project\\homework"
        traces.append({"file": path.name, "session_id": metadata["id"],
                       "records": records, "bytes": path.stat().st_size, "sha256": sha(path)})
    assert len(traces) == 10
    result["traces"] = traces
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
