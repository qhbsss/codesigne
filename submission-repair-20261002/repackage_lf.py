"""Create a fresh package whose raw ASM bytes match report program hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "homework-submit-linux-20261002"
DEST = HERE.parent / "homework-submit-linux-lf-20261002"
ZIP = DEST.with_suffix(".zip")


def h(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    assert SOURCE.is_dir()
    assert not DEST.exists() and not ZIP.exists(), "Refusing to overwrite output"
    shutil.copytree(SOURCE, DEST)
    shutil.copy2(HERE / "UPLOAD-COMPATIBILITY.md", DEST / "project/UPLOAD-COMPATIBILITY.md")
    report_raw = (DEST / "local-grade.json").read_bytes()
    report = json.loads(report_raw)
    conversions = {}
    for case, expected in report["provenance"]["program_sha256"].items():
        path = DEST / "programs" / f"{case}.asm"
        before = path.read_bytes()
        assert b"\r\n" in before and before.replace(b"\r\n", b"").find(b"\r") < 0
        after = before.replace(b"\r\n", b"\n")
        assert h(after) == expected
        path.write_bytes(after)
        conversions[case] = {
            "before_crlf_sha256": h(before), "after_lf_sha256": h(after),
            "report_program_sha256": expected, "text_content_unchanged": True,
            "crlf_replacements": before.count(b"\r\n")}
    hardware = json.loads((DEST / "hardware.json").read_text(encoding="utf-8"))
    canonical_hardware = json.dumps(hardware, sort_keys=True, separators=(",", ":")).encode()
    assert h(canonical_hardware) == report["provenance"]["hardware_sha256"]
    dump(DEST / "project/upload-artifact-check.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_unchanged_sha256": h(report_raw),
        "hardware_canonical_sha256": h(canonical_hardware),
        "hardware_report_match": True,
        "raw_zip_program_bytes_match_report": True,
        "programs": conversions,
        "server_receipt_verified": False})
    manifest_path = DEST / "project/file-manifest.json"
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p != manifest_path)
    manifest = {p.relative_to(DEST).as_posix(): {"bytes": p.stat().st_size, "sha256": h(p.read_bytes())}
                for p in files}
    dump(manifest_path, {"algorithm": "sha256", "files": manifest,
                         "excludes": ["project/file-manifest.json"]})
    with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(p for p in DEST.rglob("*") if p.is_file()):
            z.write(path, path.relative_to(DEST).as_posix())
    assert ZIP.stat().st_size < 25 * 1024**2
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert z.read("local-grade.json") == report_raw
        required = {"hardware.json", "programs/M1_P1.asm", "programs/M2_D1.asm", "local-grade.json"}
        assert required <= set(z.namelist())
        assert set(z.namelist()) == set(manifest) | {"project/file-manifest.json"}
        for case, expected in report["provenance"]["program_sha256"].items():
            assert h(z.read(f"programs/{case}.asm")) == expected
        for name, spec in manifest.items():
            assert h(z.read(name)) == spec["sha256"], name
    print(json.dumps({"zip": str(ZIP), "mib": ZIP.stat().st_size / 1024**2,
        "sha256": h(ZIP.read_bytes()), "local_score": report["experimental_score"],
        "raw_program_hashes_match_report": True, "server_verified": False}, indent=2))


if __name__ == "__main__":
    main()
