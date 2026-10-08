"""Package an already verified frozen result without modifying its raw grade."""
import argparse,json,hashlib,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
g=json.loads((a.frozen/'local-grade.json').read_text());assert g['eligible']
readme=f'''# {g['experimental_score']:.8f} 分阶段性提交包

ZIP 没有外层目录，hardware.json、programs/M1_P1.asm、programs/M2_D1.asm、local-grade.json 位于规定位置。
原始完整公共评分报告未编辑；eligible=true，官方 score=null，尚无服务端验证回执。
八万分目标{'已经达到' if g['experimental_score']>80000 else '尚未达到'}。

解压后运行 `OPENBLAS_NUM_THREADS=1 python -m project.verify_artifacts` 验证 provenance 和逐字节重生成。
按 ASSIGNMENT.md 的规则，中间提交可以不含 agent trace；截止前的最终提交须附完整 agent trace。
'''
with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    def put(name,data):
        info=zipfile.ZipInfo(name,(2026,10,9,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,data)
    for f in sorted(a.frozen.rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':put(f.relative_to(a.frozen).as_posix(),f.read_bytes())
    put('SUBMISSION.md',readme.encode())
    with zipfile.ZipFile(Path('../homework-submit-63872-20261006.zip')) as previous:
        for name in ('ASSIGNMENT.md','requirements.txt'):put(name,previous.read(name))
with zipfile.ZipFile(a.output) as z:
    assert z.testzip() is None
    for name in ('hardware.json','programs/M1_P1.asm','programs/M2_D1.asm','local-grade.json'):assert z.read(name)==(a.frozen/name).read_bytes()
digest=hashlib.sha256(a.output.read_bytes()).hexdigest();a.output.with_suffix(a.output.suffix+'.sha256').write_text(digest+'  '+a.output.name+'\n');print(a.output,digest)
