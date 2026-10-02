"""Validate and create a new ZIP beside this submission directory."""
from datetime import datetime, timezone
import json
import zipfile

from project.verify_package import ROOT, sha, validate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def included_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".zip"))


def main():
    destination = ROOT.parent / (ROOT.name + ".zip")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {destination}")
    result = validate()
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "project/package-validation.json", result)
    write_json(ROOT / "agent-trace/trace-index.json", {
        "indexed_at_utc": result["checked_at_utc"],
        "snapshot_note": "Raw files copied immediately before this package build; no later records included.",
        "sessions": result["traces"]})
    manifest_path = ROOT / "project/file-manifest.json"
    manifest = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in included_files() if p != manifest_path}
    write_json(manifest_path, {"algorithm": "sha256", "excludes": ["project/file-manifest.json"],
                               "files": manifest})
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in included_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    assert destination.stat().st_size <= 25 * 1024**2, "ZIP exceeds upload size limit"
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert {"hardware.json", "programs/M1_P1.asm", "programs/M2_D1.asm", "local-grade.json",
                "project/iteration-log.md", "agent-trace/trace-index.json"} <= names
        assert names == set(manifest) | {"project/file-manifest.json"}
        import hashlib
        for name, info in manifest.items():
            data = archive.read(name)
            assert len(data) == info["bytes"]
            assert hashlib.sha256(data).hexdigest() == info["sha256"], name
    print(json.dumps({"zip": str(destination), "bytes": destination.stat().st_size,
          "mib": destination.stat().st_size / 1024**2, "files": len(names),
          "sha256": sha(destination), "checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
