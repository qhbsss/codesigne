"""Independent statement-level hardware equations; no simulator calls."""
from math import log2

MENUS = {
    "sms": [2,4,8,12,16,20,24,28,32], "rf_kib": [16,32,64,128,256],
    "rf_tier": [1,2,3], "sh_kib": [0,32,64,96,128,192,256,512,1024],
    "sh_banks": [0,1,2,4,8,16,32,64], "sh_tc_bw": [0,32,64,128,256],
    "vector": [8,16,32,64,128], "sfu": list(range(1,129)),
    "p": [0,1,2,4,8,16,32,64], "q": [0,1,2,4,8,16,32,64],
    "hbm_channels": [1,2,4,8], "hbm_queue": [32,64,128,256,512],
    "sm_noc": [16,32,64,128,256], "global_noc": [64,128,256,512],
    "tc_count": [0,1,2,3,4,6,8], "tc_k_parallel": [1,2,4],
    "reduction_units": [0,1,2], "shared_ports": [1,2], "dma_depth": [0,1,2],
    "dma_engines": [1,2,4], "multicast": [False,True],
    "cache_mib": [0,1,2,4,8,16], "resident_groups": [1,2,4,8],
}

def area(h):
    n,r,t,s,b,bt = (h[k] for k in ("sms","rf_kib","rf_tier","sh_kib","sh_banks","sh_tc_bw"))
    v,u,p,q,c,kp=(h[k] for k in ("vector","sfu","p","q","tc_count","tc_k_parallel"))
    red,ports,z,d,g,m,C=(h[k] for k in ("reduction_units","shared_ports","dma_depth","dma_engines","resident_groups","multicast","cache_mib"))
    per=(.45+.018*v+.06*u + c*(.072+.008*(p*q+p+q)*kp)
         + .018*r*[1,1.6,2.6][t-1] + .004*s+.015*b
         + (ports-1)*(.0015*s+.008*b) + (.28+.004*bt if bt else 0)
         + red*(.10+.012*v) + .514+d*(.10+.006*(4+2*z))
         + .256+.08*(g-1)+.002*z*g+.15*(h["sm_noc"]/64)**1.3+.04*m)
    return (n*per + (3*C+.6+.012*256+.006*64 if C else 0)
            + .65*h["hbm_channels"]+.0015*h["hbm_channels"]*h["hbm_queue"]
            +1+.035*n+.8*(h["global_noc"]/128)**1.3+.2*m+2)

def validate(h):
    if set(h)!=set(MENUS): raise ValueError("hardware must contain exactly 23 known fields")
    for k,menu in MENUS.items():
        if k=="multicast":
            if type(h[k]) is not bool: raise ValueError("multicast must be bool")
        elif type(h[k]) is not int: raise ValueError(f"{k} must be integer")
        if h[k] not in menu: raise ValueError(f"invalid {k}")
    if not 1<=h["sfu"]<=h["vector"]: raise ValueError("SFU capacity")
    if h["tc_count"]==0:
        if (h["p"],h["q"],h["tc_k_parallel"])!=(0,0,1): raise ValueError("disabled TC")
    elif not h["p"] or not h["q"] or not 16<=h["p"]*h["q"]<=512: raise ValueError("TC shape")
    if not h["sh_kib"]:
        if h["sh_banks"] or h["sh_tc_bw"] or h["shared_ports"]!=1: raise ValueError("disabled SH")
    elif not h["sh_banks"] or h["sh_banks"]>h["sh_kib"]: raise ValueError("SH banks")
    if h["sh_tc_bw"] and not h["tc_count"]: raise ValueError("SH-TC interface")
    if area(h)>100: raise ValueError("area exceeds 100 AU")

def base_power(h): return .025*area(h)+.15*h["hbm_channels"]
def rf_latency(h): return 2+max(h["rf_kib"]//32,1).bit_length()-1
def sh_latency(h): return 6+max(h["sh_kib"]//(4*h["sh_banks"]),1).bit_length()-1
def rf_energy(h): return [.30,.42,.60][h["rf_tier"]-1]*(1+.1*log2(h["rf_kib"]/32))
