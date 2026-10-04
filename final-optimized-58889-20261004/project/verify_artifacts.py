"""Verify frozen grade provenance and deterministic program regeneration."""
import json,tempfile
from pathlib import Path
from .build_programs import build
from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import provenance,digest_bytes

def verify():
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'local-grade.json').read_text())
    programs={case:(root/'programs'/f'{case}.asm').read_text() for case in ('M1_P1','M2_D1')}
    hardware=Hardware.from_dict(json.loads((root/'hardware.json').read_text()))
    current=provenance(hardware,programs)
    assert current==report['provenance'], 'Source, hardware, or program provenance differs from grade'
    assert digest_bytes((root/'baseline_manifest.json').read_bytes())==report['baseline_manifest_sha256']
    assert report['eligible'] and report['experimental_score']>55811.66099137171, 'New result did not beat the prior complete grade'
    with tempfile.TemporaryDirectory(prefix='codesign-rebuild-') as temporary:
        build(Path(temporary))
        for case,text in programs.items():
            assert (Path(temporary)/f'{case}.asm').read_text()==text, f'{case} generation differs'
    print('PASS: frozen provenance, baseline, eligible improvement, and byte-identical regeneration')
    print('experimental score:', report['experimental_score'], '; 80k reached:', report['experimental_score']>80000)

if __name__=='__main__':verify()
