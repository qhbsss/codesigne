//! Trusted synthetic stress kernels. Not a student-facing grading entry point.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    hardware::Hardware,
    isa::{Arg, Instr, Op, View},
    machine::Machine,
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
    #[serde(default)]
    cycle_reference: bool,
    // Accepted for compatibility with the bounded worker. Kernels use one thread.
    #[serde(default = "one")]
    reference_threads: usize,
}
fn one() -> usize {
    1
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn run(c: &Config) -> Result<serde_json::Value> {
    c.hardware.validate()?;
    require(
        (1..=8190).contains(&c.repeats)
            && (1..=16384).contains(&c.len)
            && (1..=4096).contains(&c.stride)
            && c.reference_threads == 1,
        "invalid stress bounds/threads",
    )?;
    let mut m = Machine::new(c.hardware.clone(), true)?;
    m.timing.cycle_reference = c.cycle_reference;
    let sms = c.hardware.sms;
    let mut comparisons = 0;
    match c.kind.as_str() {
        "sfu" => {
            require(
                c.len <= 8192 && sms * (c.repeats + 2) <= 16384,
                "SFU stress exceeds ISA limits",
            )?;
            let dst = m.empty("output", sms * c.len)?;
            let mut wave = vec![vec![]; sms];
            for (sm, stream) in wave.iter_mut().enumerate() {
                stream.push(Instr::Fill {
                    dst: 0,
                    len: c.len,
                    value: 0.125,
                });
                for _ in 0..c.repeats {
                    stream.push(Instr::Vector {
                        kind: Op::Tanh,
                        dst: 0,
                        len: c.len,
                        a: Arg::reg(0),
                        b: Arg::Imm { value: 0. },
                    });
                }
                stream.push(Instr::Store {
                    tensor: dst,
                    global: View::contiguous(sm * c.len, 1, c.len),
                    local: View::contiguous(0, 1, c.len),
                });
            }
            m.wave(wave)?;
            let mut expected = 0.125f32;
            for _ in 0..c.repeats {
                expected = expected.tanh();
            }
            require(
                m.tensors[dst]
                    .data
                    .iter()
                    .all(|x| x.to_bits() == expected.to_bits()),
                "SFU result mismatch",
            )?;
            comparisons = sms * c.len;
        }
        "compute" => {
            let (rows, cols, k) = (16, 32, 256);
            let aa = 0;
            let bb = rows * k;
            let cc = bb + k * cols;
            require(
                sms * (c.repeats + 4) <= 16384,
                "compute stress exceeds wave limit",
            )?;
            let dst = m.empty("output", sms * rows * cols)?;
            let mut wave = vec![vec![]; sms];
            for (sm, stream) in wave.iter_mut().enumerate() {
                for (dst, len, value) in [
                    (aa, rows * k, 1.),
                    (bb, k * cols, 1.),
                    (cc, rows * cols, 0.),
                ] {
                    stream.push(Instr::Fill { dst, len, value });
                }
                for _ in 0..c.repeats {
                    stream.push(Instr::Mma {
                        a: aa,
                        b: bb,
                        c: cc,
                        m: rows,
                        n: cols,
                        k,
                    });
                }
                stream.push(Instr::Store {
                    tensor: dst,
                    global: View::contiguous(sm * rows * cols, 1, rows * cols),
                    local: View::contiguous(cc, 1, rows * cols),
                });
            }
            m.wave(wave)?;
            let expected = (k * c.repeats) as f32;
            require(
                m.tensors[dst].data.iter().all(|x| *x == expected),
                "compute result mismatch",
            )?;
            comparisons = sms * rows * cols;
        }
        "dma" => {
            require(
                sms * 2 * c.repeats <= 16384,
                "DMA stress exceeds wave limit",
            )?;
            let source_len = (c.len - 1) * c.stride + 1;
            let src = m.alloc("input", (0..source_len).map(|i| (i % 251) as f32).collect())?;
            let dst = m.empty("output", sms * c.len)?;
            let mut wave = vec![vec![]; sms];
            for (sm, stream) in wave.iter_mut().enumerate() {
                for _ in 0..c.repeats {
                    stream.push(Instr::Load {
                        tensor: src,
                        global: View {
                            base: 0,
                            rows: c.len,
                            cols: 1,
                            row_stride: c.stride,
                            col_stride: 1,
                        },
                        local: View::contiguous(0, c.len, 1),
                    });
                    stream.push(Instr::Store {
                        tensor: dst,
                        global: View::contiguous(sm * c.len, 1, c.len),
                        local: View::contiguous(0, 1, c.len),
                    });
                }
            }
            m.wave(wave)?;
            for (i, x) in m.tensors[dst].data.iter().enumerate() {
                require(
                    *x == ((i % c.len * c.stride) % 251) as f32,
                    "DMA result mismatch",
                )?;
            }
            comparisons = sms * c.len;
            let r = m.report();
            require(
                r.stats.max_hbm_inflight <= c.hardware.hbm_channels * c.hardware.hbm_queue,
                "HBM inflight exceeded hardware",
            )?;
            require(
                r.stats.max_frontend <= 32 && r.stats.max_write_staged <= 2,
                "frontend/write staging unbounded",
            )?;
        }
        "shared" => {
            require(
                sms * (2 * c.repeats + 2) <= 16384,
                "SH stress exceeds wave limit",
            )?;
            let src = m.alloc("input", (0..c.len).map(|i| (i % 251) as f32).collect())?;
            let dst = m.empty("output", sms * c.len)?;
            let mut wave = vec![vec![]; sms];
            for (sm, stream) in wave.iter_mut().enumerate() {
                let shared = View {
                    base: 0,
                    rows: c.len,
                    cols: 1,
                    row_stride: c.stride,
                    col_stride: 1,
                };
                let local = View::contiguous(0, c.len, 1);
                stream.push(Instr::LoadShared {
                    tensor: src,
                    global: View::contiguous(0, c.len, 1),
                    local: shared,
                });
                for _ in 0..c.repeats {
                    stream.push(Instr::SharedRead { shared, local });
                    stream.push(Instr::SharedWrite { shared, local });
                }
                stream.push(Instr::Store {
                    tensor: dst,
                    global: View::contiguous(sm * c.len, c.len, 1),
                    local,
                });
            }
            m.wave(wave)?;
            for (i, x) in m.tensors[dst].data.iter().enumerate() {
                require(*x == ((i % c.len) % 251) as f32, "SH result mismatch")?;
            }
            comparisons = sms * c.len;
        }
        "allocation_guard" => {
            let before = m.report().hbm_allocated_bytes;
            require(
                m.empty("overflow", 2 * 1024 * 1024 * 1024 / 4 + 1).is_err(),
                "HBM cap not enforced",
            )?;
            require(
                m.report().hbm_allocated_bytes == before,
                "failed allocation changed capacity",
            )?;
        }
        "metadata_guard" => {
            let n = 16384;
            let tensor = m.alloc("io", vec![1.; sms * n])?;
            let mut wave = vec![vec![]; sms];
            for (sm, stream) in wave.iter_mut().enumerate() {
                let global = View::contiguous(sm * n, n, 1);
                let local = View::contiguous(0, n, 1);
                stream.push(Instr::Load {
                    tensor,
                    global,
                    local,
                });
                stream.push(Instr::Store {
                    tensor,
                    global,
                    local,
                });
            }
            let err = m.wave(wave).err().ok_or("metadata cap not enforced")?;
            require(err.contains("access intervals exceed"), &err)?;
            require(m.report().stats.cycles == 0, "rejected wave changed timing")?;
        }
        _ => return Err("unknown stress kind".into()),
    }
    let report = m.report();
    Ok(
        serde_json::json!({"comparison":{"elements":comparisons,"method":"all written elements vs analytic oracle; guards check rejection"},"report":report}),
    )
}
fn main() {
    if let Err(e) = entry() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn entry() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    require(
        args.len() == 5 && args[1] == "check",
        "extreme_probe check CONFIG SEED NEW_OUTPUT",
    )?;
    let path = Path::new(&args[2]);
    require(
        fs::metadata(path).map_err(|e| e.to_string())?.len() <= 1024 * 1024,
        "config too large",
    )?;
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let config: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let out = Path::new(&args[4]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let result = run(&config);
    let value = serde_json::json!({"status":if result.is_ok(){"complete"}else{"failed"},"error":result.as_ref().err(),"evaluation":result.as_ref().ok(),
        "source_sha256":vnext_sim::source::hash(),"probe_sha256":format!("{:x}",Sha256::digest(include_bytes!("extreme_probe.rs"))),
        "config_sha256":format!("{:x}",Sha256::digest(&bytes)),"host_seconds":start.elapsed().as_secs_f64(),"official_score":null});
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    result.map(|_| ())
}
