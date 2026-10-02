"""Read-only artifact/release/manifest verification; no expensive grade rerun."""
import hashlib
import json
from pathlib import Path


def h(data):
    return hashlib.sha256(data).hexdigest()


def jh(value):
    return h(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "project/file-manifest.json").read_text(encoding="utf-8"))
    for name, spec in manifest["files"].items():
        data = (root / name).read_bytes()
        assert h(data) == spec["sha256"] and len(data) == spec["bytes"], name
    report = json.loads((root / "local-grade.json").read_text(encoding="utf-8"))
    p = report["provenance"]
    assert report["eligible"] is True
    assert report["seed_sha256"] == [h(b"challenge-seed-v1:7")]
    assert jh(p["source_sha256"]) == p["simulator_sha256"]
    assert all("\\" not in name for name in p["source_sha256"])
    for name, digest in p["source_sha256"].items():
        assert h((root / name).read_bytes()) == digest, name
    assert jh(json.loads((root / "hardware.json").read_text())) == p["hardware_sha256"]
    for case, digest in p["program_sha256"].items():
        text = (root / "programs" / (case + ".asm")).read_text(encoding="utf-8")
        assert h(text.encode()) == digest
    assert h((root / "baseline_manifest.json").read_bytes()) == report["baseline_manifest_sha256"]
    print(json.dumps({"passed": True, "manifest_files": len(manifest["files"]),
        "local_score": report["experimental_score"], "simulator_sha256": p["simulator_sha256"],
        "server_verified": False}, indent=2))


if __name__ == "__main__":
    main()
