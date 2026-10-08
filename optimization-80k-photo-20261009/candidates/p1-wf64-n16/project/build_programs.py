"""Regenerate both graded ASM artifacts, including deterministic P1 compaction."""
from pathlib import Path
from hashlib import sha256
from .schedule import generate_m1_p1,generate_m1_d1
from .compact_names import compact

def build(output=Path('programs')):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,generator in [('M1_P1',generate_m1_p1),('M2_D1',generate_m1_d1)]:
        text,_,_=generator()
        if name=='M1_P1':
            from .strip_tail_prefetch import strip
            text=compact(strip(text))
        from .constant_fold import compact as fold_addresses
        text=fold_addresses(text)
        data=text.encode();path=output/(name+'.asm');path.write_bytes(data)
        print(name,len(data),sha256(data).hexdigest())

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('programs'))
    build(parser.parse_args().output)
