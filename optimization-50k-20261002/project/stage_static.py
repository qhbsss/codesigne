"""Print per-stage static demand for the accepted P1 literal program."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codesign.challenge.hardware import Hardware

from .rough_model import profile_program


def main() -> None:
    import json

    root = Path("checkpoints/accepted-28885")
    hardware = Hardware.from_dict(json.loads((root / "hardware.json").read_text()))
    text = (root / "programs/M1_P1.asm").read_text()
    begins = [line for line in text.splitlines() if line.startswith("WG.BEGIN ")]
    ends = [line for line in text.splitlines() if line.startswith("WG.END ")]
    pieces = text.split("BARRIER ")
    # Each barrier's JSON payload is the first line of the following piece;
    # operations preceding barrier i are in pieces[i].
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    names = ("ln1", "qkv_attn", "wo", "ln2", "w1", "w2")
    for index, piece in enumerate(pieces[:-1]):
        lines = piece.splitlines()
        if index:
            lines = lines[1:]
        lines = [
            line
            for line in lines
            if line and not line.startswith(("WG.BEGIN ", "WG.END ", "STEP.COMMIT "))
        ]
        mini = "\n".join(begins + lines + ends) + "\n"
        profile = profile_program(hardware, mini)
        phase = "prompt" if index < 36 else "step"
        stage = names[index % 6]
        item = totals[(phase, stage)]
        item["instructions"] += profile.instruction_count
        item["read_lines"] += profile.hbm_read_line_requests
        item["write_lines"] += profile.hbm_write_line_requests
        item["noc_lines"] += profile.shared_demands["noc_line_slices"]
        item["tc"] += max(d.get("tc", 0) for d in profile.sm_demands.values())
        item["rf_read"] += max(d.get("rf_read", 0) for d in profile.sm_demands.values())
        item["rf_write"] += max(d.get("rf_write", 0) for d in profile.sm_demands.values())
        item["vec"] += max(d.get("vec", 0) for d in profile.sm_demands.values())
    for phase in ("prompt", "step"):
        print(phase)
        for stage in names:
            print(f"  {stage:9s} {dict(totals[(phase, stage)])}")


if __name__ == "__main__":
    main()
