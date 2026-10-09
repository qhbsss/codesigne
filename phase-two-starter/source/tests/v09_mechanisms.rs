#![cfg(feature = "concurrent")]
use vnext_sim::{
    isa::{Arg, Instr, Op, View},
    v09_hardware::Hardware,
    v09_schedule::{Command, Group, Report, Scheduler},
};
fn hardware() -> Hardware {
    let mut h = Hardware::default();
    h.base.sms = 2;
    h.base.sfu = 1;
    h.resident_groups = 2;
    h
}
fn group(commands: Vec<Command>) -> Group {
    Group {
        sm: 0,
        rf_kib: 8,
        sh_kib: 0,
        commands,
    }
}
fn load(token: u32, local: usize) -> Command {
    Command::Async {
        token,
        instruction: Instr::Load {
            tensor: 0,
            global: View::contiguous(local, 1, 16),
            local: View::contiguous(local, 1, 16),
        },
    }
}
fn compute() -> Command {
    Command::Run {
        instruction: Instr::Vector {
            kind: Op::Tanh,
            dst: 256,
            len: 128,
            a: Arg::Imm { value: 0.1 },
            b: Arg::Imm { value: 0. },
        },
    }
}
#[test]
fn recovered_knobs_have_measured_mechanism_effects() {
    fn run(h: Hardware, groups: &[Group]) -> Report {
        let mut s = Scheduler::new(h, false).unwrap();
        s.wave(groups, &[0]).unwrap();
        s.report()
    }
    let mut evidence = vec![];
    let mma = group(vec![Command::Run {
        instruction: Instr::Mma {
            a: 0,
            b: 512,
            c: 1024,
            m: 8,
            n: 8,
            k: 64,
        },
    }]);
    let h = hardware();
    let one = run(h.clone(), &[mma.clone(), mma.clone()]);
    let mut two_h = h.clone();
    two_h.tc_count = 2;
    let two = run(two_h.clone(), &[mma.clone(), mma.clone()]);
    assert!(two.stats.cycles < one.stats.cycles);
    assert_eq!(one.stats.rf_bytes, two.stats.rf_bytes);
    evidence.push(serde_json::json!({"knob":"tc_count 1->2","cycles":[one.stats.cycles,two.stats.cycles],"rf_bytes":[one.stats.rf_bytes,two.stats.rf_bytes]}));
    let k1 = run(h.clone(), std::slice::from_ref(&mma));
    let mut k4h = h.clone();
    k4h.tc_k_parallel = 4;
    let k4 = run(k4h, std::slice::from_ref(&mma));
    assert!(k4.stats.cycles < k1.stats.cycles);
    assert_eq!(k1.stats.physical_fmas, k4.stats.physical_fmas);
    evidence.push(serde_json::json!({"knob":"tc_k_parallel 1->4","cycles":[k1.stats.cycles,k4.stats.cycles],"physical_fmas":[k1.stats.physical_fmas,k4.stats.physical_fmas]}));
    let reduce = group(vec![Command::Run {
        instruction: Instr::Reduce {
            src: 0,
            dst: 512,
            scratch: 1024,
            len: 128,
            max: false,
        },
    }]);
    let mut reductions = vec![];
    for units in [0, 1, 2] {
        let mut hh = h.clone();
        hh.reduction_units = units;
        reductions.push(run(hh, &[reduce.clone(), reduce.clone()]).stats.cycles);
    }
    assert!(reductions[1] < reductions[0]);
    assert!(reductions[2] < reductions[1]);
    evidence.push(serde_json::json!({"knob":"reduction_units 0->1->2","cycles":reductions}));
    let mut sh = group(vec![Command::Run {
        instruction: Instr::SharedRead {
            shared: View {
                base: 0,
                rows: 16,
                cols: 1,
                row_stride: 8,
                col_stride: 1,
            },
            local: View::contiguous(0, 1, 16),
        },
    }]);
    sh.sh_kib = 4;
    let mut shh = h.clone();
    shh.base.sh_kib = 64;
    shh.base.sh_banks = 8;
    let sh1 = run(shh.clone(), std::slice::from_ref(&sh));
    shh.shared_ports = 2;
    let sh2 = run(shh, &[sh]);
    assert!(sh2.stats.cycles < sh1.stats.cycles);
    evidence.push(serde_json::json!({"knob":"shared_ports 1->2","cycles":[sh1.stats.cycles,sh2.stats.cycles],"sh_bytes":[sh1.stats.sh_bytes,sh2.stats.sh_bytes]}));
    let transfer = group(vec![load(1, 0), Command::Wait { tokens: vec![1] }]);
    let mut dh = h.clone();
    dh.dma_depth = 1;
    let d1 = run(dh.clone(), &[transfer.clone(), transfer.clone()]);
    dh.dma_engines = 2;
    let d2 = run(dh, &[transfer.clone(), transfer.clone()]);
    assert!(d2.stats.cycles < d1.stats.cycles);
    evidence.push(serde_json::json!({"knob":"dma_engines 1->2","cycles":[d1.stats.cycles,d2.stats.cycles],"hbm_reads":[d1.memory.hbm_reads,d2.memory.hbm_reads]}));
    let queued = vec![
        group(vec![
            load(1, 0),
            load(2, 16),
            compute(),
            Command::Wait { tokens: vec![1, 2] }
        ]);
        2
    ];
    let mut dh = h.clone();
    dh.dma_depth = 1;
    let depth1 = run(dh.clone(), &queued);
    dh.dma_depth = 2;
    let depth2 = run(dh, &queued);
    assert_eq!(depth1.stats.max_descriptors_per_engine, 2);
    assert_eq!(depth2.stats.max_descriptors_per_engine, 3);
    evidence.push(serde_json::json!({"knob":"dma_depth 1->2","cycles":[depth1.stats.cycles,depth2.stats.cycles],"max_descriptors_per_engine":[depth1.stats.max_descriptors_per_engine,depth2.stats.max_descriptors_per_engine]}));
    let reads = group(
        (0..4)
            .map(|_| Command::Run {
                instruction: Instr::Load {
                    tensor: 0,
                    global: View::contiguous(0, 1, 16),
                    local: View::contiguous(0, 1, 16),
                },
            })
            .collect(),
    );
    let bypass = run(h.clone(), std::slice::from_ref(&reads));
    let mut ch = h.clone();
    ch.cache_mib = 1;
    let cached = run(ch, &[reads]);
    assert_eq!(bypass.memory.hbm_reads, 4);
    assert_eq!(cached.memory.hbm_reads, 1);
    assert_eq!(cached.memory.cache_hits, 3);
    evidence.push(serde_json::json!({"knob":"cache_mib 0->1","cycles":[bypass.stats.cycles,cached.stats.cycles],"hbm_reads":[bypass.memory.hbm_reads,cached.memory.hbm_reads],"cache_hits":cached.memory.cache_hits}));
    let mut broadcast = vec![transfer.clone(), transfer];
    broadcast[1].sm = 1;
    let mut mh = h.clone();
    mh.dma_depth = 1;
    let unicast = run(mh.clone(), &broadcast);
    mh.multicast = true;
    let multicast = run(mh, &broadcast);
    assert!(multicast.memory.hbm_reads < unicast.memory.hbm_reads);
    assert_eq!(
        multicast.memory.noc_endpoint_bytes,
        unicast.memory.noc_endpoint_bytes
    );
    evidence.push(serde_json::json!({"knob":"multicast false->true","cycles":[unicast.stats.cycles,multicast.stats.cycles],"hbm_reads":[unicast.memory.hbm_reads,multicast.memory.hbm_reads],"noc_endpoint_bytes":[unicast.memory.noc_endpoint_bytes,multicast.memory.noc_endpoint_bytes],"noc_global_bytes":[unicast.memory.noc_global_bytes,multicast.memory.noc_global_bytes]}));
    two_h.resident_groups = 1;
    let mut serial = Scheduler::new(two_h, false).unwrap();
    serial.wave(std::slice::from_ref(&mma), &[0]).unwrap();
    serial.wave(std::slice::from_ref(&mma), &[0]).unwrap();
    assert!(two.stats.cycles < serial.stats.cycles);
    evidence.push(serde_json::json!({"knob":"resident_groups 1->2","cycles":[serial.stats.cycles,two.stats.cycles],"max_resident_groups":[serial.stats.max_resident_groups,two.stats.max_resident_groups]}));
    println!(
        "KNOB_EVIDENCE={}",
        serde_json::to_string(&evidence).unwrap()
    );
}
