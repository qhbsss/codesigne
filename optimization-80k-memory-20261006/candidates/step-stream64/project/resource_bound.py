"""Safe work-conservation lower bounds for a fixed hardware and instruction graph.

Sum max resource demand between barriers covering all live groups. Ignore
latencies, hazards, power and source HBM demand, so this is optimistic.
"""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.service import mma_service,vector_cycles,sfu_cycles,reduction_cycles
from .rough_model import _line_ids
p=argparse.ArgumentParser();p.add_argument('--hardware',required=True);p.add_argument('--program',required=True);p.add_argument('--report',required=True);a=p.parse_args()
h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));ports={'2R1W':(2,1),'4R2W':(4,2),'8R4W':(8,4)}[h.rf_ports]
live={};sm_of={};work=defaultdict(float);segments=[]
def add(sm,key,n):work[(sm,key)]+=n
def finish():
 if work:
  resource,value=max(work.items(),key=lambda kv:kv[1]);segments.append({'bound_cycles':math.ceil(value),'limiting_resource':str(resource),'work':{str(k):v for k,v in work.items()}});work.clear()
for ins in iter_parse(Path(a.program).read_text()):
 op,d=ins.op,ins.args
 if op=='WG.BEGIN':live[d['wg']]=d['sm'];sm_of[d['wg']]=d['sm'];continue
 if op=='WG.END':live.pop(d['wg'],None);continue
 if op=='BARRIER':
  if set(d.get('wgs',[]))==set(live):finish()
  continue
 if op in ('WAIT','STEP.COMMIT'):continue
 if op in ('LD','ST'):
  src,dst=d['src'],d['dst'];sm=sm_of[src['wg'] if src['space']!='HBM' else dst['wg']];payload=src['count']*4
  if src['space']=='RF':add(sm,'rf_read',math.ceil(payload/(16*ports[0])))
  if dst['space']=='RF':add(sm,'rf_write',math.ceil(payload/(16*ports[1])))
  if 'HBM' in (src['space'],dst['space']):
   view=src if src['space']=='HBM' else dst;lines=len(_line_ids(view));add('global','noc',lines*64/h.noc_bytes_per_cycle)
   add(sm,'noc_in' if src['space']=='HBM' else 'noc_out',lines*64/h.sm_noc_bytes_per_cycle);add(sm,'dma',lines/h.dma_engines)
  continue
 sm=sm_of[d['acc']['wg'] if op=='MMA.ACC' else d['dst']['wg']]
 if op=='MMA.ACC':
  demand=mma_service(h,d['m'],d['n'],d['k']);rb,wb=demand.rf_read_bytes,demand.rf_write_bytes;add(sm,'tc',demand.compute_cycles)
 elif op=='VEC':
  rb=4*sum(v['count'] for v in d['src'] if 'space' in v);wb=4*d['dst']['count'];add(sm,'vec',vector_cycles(d['dst']['count'],h.vector_lanes))
 elif op=='SFU':
  rb=wb=4*d['src']['count'];add(sm,'sfu',sfu_cycles(d['src']['count'],h.sfu_lanes))
 elif op=='REDUCE':
  rb=4*d['src']['count'];wb=4;add(sm,'reduce' if h.reduction_units else 'vec',reduction_cycles(d['src']['count'],h.vector_lanes,h.reduction_units))
 else:raise ValueError(op)
 add(sm,'rf_read',math.ceil(rb/(16*ports[0])));add(sm,'rf_write',math.ceil(wb/(16*ports[1])))
finish();result={'kind':'fixed-instruction-resource-lower-bound','lower_bound_cycles':sum(s['bound_cycles'] for s in segments),'segments':segments,'limitations':'Not a bound for redesigned graphs or different hardware; omits latency, hazards, SH service and power.'}
Path(a.report).write_text(json.dumps(result,indent=2)+'\n');print('fixed graph lower bound:',result['lower_bound_cycles'])
