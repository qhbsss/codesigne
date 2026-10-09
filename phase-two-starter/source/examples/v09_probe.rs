//! Trusted concurrent stress probes with exhaustive analytical output checks.
//! Diagnostic only: a numerically valid probe may exceed architectural power.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    isa::{Arg, Instr, Op, View},
    machine::Machine,
    v09_hardware::Hardware,
    v09_schedule::{Command, Group},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    kind: String,
    hardware: Hardware,
    #[serde(default = "one")]
    repeats: usize,
    #[serde(default = "one")]
    len: usize,
    #[serde(default = "one")]
    stride: usize,
    #[serde(default = "one")]
    reference_threads: usize,
}
fn one() -> usize {
    1
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn run(instruction: Instr) -> Command {
    Command::Run { instruction }
}
fn loads(
    commands: &mut Vec<Command>,
    instructions: Vec<Instr>,
    asynchronous: bool,
    next_token: &mut u32,
) {
    let mut tokens = vec![];
    for instruction in instructions {
        if asynchronous {
            let token = *next_token;
            *next_token += 1;
            tokens.push(token);
            commands.push(Command::Async { token, instruction });
        } else {
            commands.push(run(instruction));
        }
    }
    if !tokens.is_empty() {
        commands.push(Command::Wait { tokens });
    }
}
fn evaluate(c: &Config) -> Result<serde_json::Value> {
    c.hardware.validate()?;
    require(
        (1..=8190).contains(&c.repeats)
            && (1..=16384).contains(&c.len)
            && (1..=4096).contains(&c.stride)
            && c.reference_threads == 1,
        "invalid probe bounds",
    )?;
    let h = &c.hardware;
    let count = h.base.sms * h.resident_groups;
    let mut m = Machine::new_concurrent(h.clone(), true, 1, false)?;
    let mut compared = 0;
    match c.kind.as_str() {
        "all_units" | "all_units_rf" | "all_units_tail" => {
            require(h.tc_count > 0, "all_units requires TC")?;
            let k = if c.kind == "all_units_tail" { 31 } else { 32 };
            let shared = c.kind != "all_units_rf" && h.base.sh_kib > 0 && h.base.sh_tc_bw > 0;
            let (rf_kib, sh_kib) = (8, if shared { 4 } else { 0 });
            require(
                rf_kib * h.resident_groups <= h.base.rf_kib
                    && sh_kib * h.resident_groups <= h.base.sh_kib,
                "insufficient group reservation for all_units",
            )?;
            require(
                count * (c.repeats + 12) <= 16384,
                "all_units exceeds wave instruction limit",
            )?;
            let src = m.alloc("shared_readonly_ones", vec![1.; 512])?;
            let dst = m.empty("disjoint_outputs", count * 256)?;
            let mut groups = vec![];
            for index in 0..count {
                let mut commands = vec![];
                let mut token = 0;
                let aa = Instr::Load {
                    tensor: src,
                    global: View::contiguous(0, 16, k),
                    local: View::contiguous(0, 16, k),
                };
                let bb = if shared {
                    Instr::LoadShared {
                        tensor: src,
                        global: View::contiguous(0, k, 16),
                        local: View::contiguous(0, k, 16),
                    }
                } else {
                    Instr::Load {
                        tensor: src,
                        global: View::contiguous(0, k, 16),
                        local: View::contiguous(512, k, 16),
                    }
                };
                loads(&mut commands, vec![aa, bb], h.dma_depth > 0, &mut token);
                commands.push(run(Instr::Fill {
                    dst: 1024,
                    len: 256,
                    value: 0.,
                }));
                for _ in 0..c.repeats {
                    commands.push(run(if shared {
                        Instr::MmaShared {
                            a: 0,
                            b: View::contiguous(0, k, 16),
                            c: 1024,
                            m: 16,
                            n: 16,
                            k,
                        }
                    } else {
                        Instr::Mma {
                            a: 0,
                            b: 512,
                            c: 1024,
                            m: 16,
                            n: 16,
                            k,
                        }
                    }));
                }
                // Exercise SFU, reduction and vector resources without changing
                // the exactly representable integer MMA output oracle.
                commands.push(run(Instr::Fill {
                    dst: 1280,
                    len: 128,
                    value: 0.,
                }));
                commands.push(run(Instr::Vector {
                    kind: Op::Tanh,
                    dst: 1280,
                    len: 128,
                    a: Arg::reg(1280),
                    b: Arg::Imm { value: 0. },
                }));
                commands.push(run(Instr::Reduce {
                    scratch: 1536,
                    dst: 1792,
                    src: 1280,
                    len: 128,
                    max: false,
                }));
                commands.push(run(Instr::Vector {
                    kind: Op::Add,
                    dst: 1024,
                    len: 256,
                    a: Arg::reg(1024),
                    b: Arg::scalar(1792),
                }));
                commands.push(run(Instr::Store {
                    tensor: dst,
                    global: View::contiguous(index * 256, 1, 256),
                    local: View::contiguous(1024, 1, 256),
                }));
                groups.push(Group {
                    sm: index / h.resident_groups,
                    rf_kib,
                    sh_kib,
                    commands,
                });
            }
            m.concurrent_wave(groups)?;
            let expected = (k * c.repeats) as f32;
            require(
                m.tensors[dst].data.iter().all(|x| *x == expected),
                "all_units output mismatch",
            )?;
            compared = count * 256;
        }
        "dma_strided" => {
            let rf_kib = (c.len * 4).div_ceil(1024).max(1);
            require(
                rf_kib * h.resident_groups <= h.base.rf_kib,
                "insufficient per-group RF for DMA length",
            )?;
            require(
                count * c.repeats * 2 <= 16384,
                "DMA exceeds wave instruction limit",
            )?;
            let length = (c.len - 1) * c.stride + 1;
            let src = m.alloc(
                "shared_strided_input",
                (0..length).map(|i| (i % 251) as f32).collect(),
            )?;
            let dst = m.empty("disjoint_outputs", count * c.len)?;
            let mut groups = vec![];
            for index in 0..count {
                let mut commands = vec![];
                let mut token = 0;
                for _ in 0..c.repeats {
                    loads(
                        &mut commands,
                        vec![Instr::Load {
                            tensor: src,
                            global: View {
                                base: 0,
                                rows: c.len,
                                cols: 1,
                                row_stride: c.stride,
                                col_stride: 1,
                            },
                            local: View::contiguous(0, c.len, 1),
                        }],
                        h.dma_depth > 0,
                        &mut token,
                    );
                    commands.push(run(Instr::Store {
                        tensor: dst,
                        global: View::contiguous(index * c.len, 1, c.len),
                        local: View::contiguous(0, 1, c.len),
                    }));
                }
                groups.push(Group {
                    sm: index / h.resident_groups,
                    rf_kib,
                    sh_kib: 0,
                    commands,
                });
            }
            m.concurrent_wave(groups)?;
            require(
                m.tensors[dst]
                    .data
                    .iter()
                    .enumerate()
                    .all(|(i, x)| *x == ((i % c.len * c.stride) % 251) as f32),
                "strided DMA output mismatch",
            )?;
            compared = count * c.len;
        }
        "allocation_guard" => {
            let before = m.hbm_usage();
            require(
                m.empty("must_reject", 2 * 1024 * 1024 * 1024 / 4 + 1)
                    .is_err(),
                "HBM cap not enforced",
            )?;
            require(
                m.hbm_usage() == before,
                "failed allocation changed capacity",
            )?;
        }
        "invalid_async_guard" => {
            let groups = vec![Group {
                sm: 0,
                rf_kib: 4,
                sh_kib: 0,
                commands: vec![Command::Wait { tokens: vec![999] }],
            }];
            require(
                m.concurrent_wave(groups).is_err(),
                "unknown wait token accepted",
            )?;
        }
        _ => return Err("unknown probe kind".into()),
    }
    let report = m.concurrent_report()?;
    require(
        report.memory.max_hbm_inflight <= h.base.hbm_channels * h.base.hbm_queue,
        "HBM Q exceeded",
    )?;
    require(
        report.memory.max_pending <= h.base.sms * 128,
        "frontend capacity exceeded",
    )?;
    require(
        report.memory.max_cache_mshrs <= h.cache_mshrs(),
        "cache MSHRs exceeded",
    )?;
    require(
        report.stats.max_resident_groups <= h.resident_groups,
        "resident group limit exceeded",
    )?;
    Ok(
        serde_json::json!({"comparison":{"elements":compared,"max_abs":0.,"max_scaled":0.},"report":report,"hbm_peak_allocated_bytes":m.hbm_usage().1,"guard_pass":true,"groups":count}),
    )
}
fn main() {
    if let Err(error) = cli() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn cli() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || args[1] != "check" {
        return Err("v09_probe check CONFIG SEED NEW_OUTPUT_DIR".into());
    }
    let seed = args[3].parse::<u64>().map_err(|_| "invalid seed")?;
    let input = Path::new(&args[2]);
    let out = Path::new(&args[4]);
    if fs::metadata(input).map_err(|e| e.to_string())?.len() > 1024 * 1024 {
        return Err("config too large".into());
    }
    let bytes = fs::read(input).map_err(|e| e.to_string())?;
    let c: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let result = evaluate(&c);
    let output = serde_json::json!({"status":if result.is_ok(){"complete"}else{"failed"},"evaluation":result.as_ref().ok(),"error":result.as_ref().err(),"host_seconds":start.elapsed().as_secs_f64(),"source_sha256":vnext_sim::source::hash(),"probe_sha256":format!("{:x}",Sha256::digest(include_bytes!("v09_probe.rs"))),"config_sha256":format!("{:x}",Sha256::digest(&bytes)),"seed":seed,"official_score":null});
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    result.map(|_| ())
}
