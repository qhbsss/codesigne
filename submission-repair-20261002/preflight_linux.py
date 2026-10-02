"""Read-only release fingerprint preflight using unmodified official modules."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent / "official-linux"
sys.path.insert(0, str(root))
from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import provenance
import numpy as np

hardware = Hardware.from_dict(json.loads((root / "hardware.json").read_text()))
programs = {c: (root / "programs" / (c + ".asm")).read_text() for c in ("M1_P1", "M2_D1")}
old = json.loads((root.parent.parent / "homework-complete-28885-20261002/local-grade.json").read_text())
current = provenance(hardware, programs)
assert current["source_sha256"] == {k.replace("\\", "/"): v for k, v in old["provenance"]["source_sha256"].items()}
for key in ("hardware_sha256", "program_sha256", "cost_sha256", "workload_sha256"):
    assert current[key] == old["provenance"][key], key
print(json.dumps({"python": sys.version, "numpy": np.__version__,
                  "only_provenance_difference": "source path separators and their aggregate hash",
                  "old_windows_fingerprint": old["provenance"]["simulator_sha256"],
                  "official_linux_fingerprint": current["simulator_sha256"]}, indent=2))
