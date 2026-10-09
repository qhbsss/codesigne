//! Exact timing transition backend. Reuses the unmodified published machine;
//! deliberately never claims to independently prove functional correctness.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path, time::Instant};
use vnext_sim::{
    Result, compact,
    isa::Instr,
    v09_hardware::Hardware,
    v09_schedule::{Group, Scheduler},
    v09_submission::{CONTRACT, Record},
};

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("model-runtime <predict|profiles|wave> INPUT OUTPUT.json".into());
    }
    if Path::new(&a[3]).exists() {
        return Err("output already exists".into());
    }
    let start = Instant::now();
    let result = match a[1].as_str() {
        "predict" => predict(Path::new(&a[2]))?,
        "profiles" => {
            let v: Value = serde_json::from_slice(&fs::read(&a[2]).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let h: Hardware =
                serde_json::from_value(v["hardware"].clone()).map_err(|e| e.to_string())?;
            h.validate()?;
            let mut profiles = vec![];
            for item in v["instructions"].as_array().ok_or("missing instructions")? {
                let i: Instr = serde_json::from_value(item.clone()).map_err(|e| e.to_string())?;
                profiles
                    .push(serde_json::to_value(h.compute_profile(&i)?).map_err(|e| e.to_string())?);
            }
            json!({"area_au":h.area(),"base_power_w":h.base_power(),"profiles":profiles})
        }
        "wave" => {
            let v: Value = serde_json::from_slice(&fs::read(&a[2]).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let h: Hardware =
                serde_json::from_value(v["hardware"].clone()).map_err(|e| e.to_string())?;
            let groups: Vec<Group> =
                serde_json::from_value(v["groups"].clone()).map_err(|e| e.to_string())?;
            let bases: Vec<usize> =
                serde_json::from_value(v.get("bases").cloned().unwrap_or(json!([])))
                    .map_err(|e| e.to_string())?;
            let mut s = Scheduler::new(h, false)?;
            s.wave(&groups, &bases)?;
            json!({"report":s.report(),"numerical_correctness":"not_checked"})
        }
        _ => return Err("unknown mode".into()),
    };
    let envelope = json!({"backend":"shared_official_transition_core",
        "model":vnext_sim::MODEL,"contract":CONTRACT,
        "source_sha256":vnext_sim::source::hash(),
        "runtime_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("main.rs"))),
        "host_seconds":start.elapsed().as_secs_f64(),"result":result});
    fs::write(
        &a[3],
        serde_json::to_vec_pretty(&envelope).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
struct Allocation {
    base: usize,
    len: usize,
    input: bool,
}
fn input_len(
    source: &str,
    h: &vnext_sim::v09_submission::Header,
    step: usize,
) -> Result<(usize, usize)> {
    let s = h.shape;
    if source == format!("input{step}") {
        let n = h.batch * if h.decode { 1 } else { s.prefill } * s.d;
        return Ok((n, n));
    }
    for l in 0..s.layers {
        let prefix = format!("layer{l}.");
        if let Some(name) = source.strip_prefix(&prefix) {
            let n = match name {
                "qkv" => s.d * 3 * s.d,
                "qb" => 3 * s.d,
                "out" => s.d * s.d,
                "up" => s.d * s.f,
                "ub" => s.f,
                "down" => s.f * s.d,
                "ob" | "db" | "g1" | "b1" | "g2" | "b2" => s.d,
                "past_k" | "past_v" if h.decode => {
                    return Ok((
                        s.history * h.batch * s.d,
                        (s.history + s.steps) * h.batch * s.d,
                    ));
                }
                _ => return Err(format!("invalid input source {source}")),
            };
            return Ok((n, n));
        }
    }
    Err(format!("unavailable or noncanonical input source {source}"))
}
fn checked_view(tensor: usize, base: usize, len: usize, allocs: &[Allocation]) -> Result<()> {
    let a = allocs.get(tensor).ok_or("tensor does not exist")?;
    if base.checked_add(len).ok_or("view overflow")? > a.len {
        return Err("view out of bounds".into());
    }
    Ok(())
}
fn predict(path: &Path) -> Result<Value> {
    let mut reader = compact::Reader::open(path)?;
    let h = reader.header.clone();
    if h.model != vnext_sim::MODEL || h.contract != CONTRACT {
        return Err("wrong model/contract".into());
    }
    // Validate shape/profile admission without allocating fixture numerical values.
    let cfg = vnext_sim::v09_submission::Config {
        shape: h.shape,
        batch: h.batch,
        decode: h.decode,
        hardware: h.hardware.clone(),
        policy: vnext_sim::baseline::Policy::square(16),
        fixture_profile: vnext_sim::baseline::FixtureProfile::AttentionStress,
        reference_threads: 1,
        parallel_groups: 1,
        async_prefetch: false,
    };
    cfg.validate()?;
    let mut sched = Scheduler::new(h.hardware.clone(), false)?;
    // The official non-functional pass validates global races, aliases and
    // shape/budget guards omitted by Scheduler alone. Placeholder tensor
    // storage is allocated, but no fixture/reference or FP32 operations run.
    let mut static_checker =
        vnext_sim::machine::Machine::new_concurrent(h.hardware.clone(), false, 1, false)?;
    let mut allocs: Vec<Allocation> = vec![];
    let mut used = HashSet::new();
    let (mut live, mut peak, mut cumulative) = (0usize, 0usize, 0u64);
    let (mut step, mut records, mut primitives, mut work) = (0usize, 0usize, 0u64, 0u64);
    let steps = if h.decode { h.shape.steps } else { 1 };
    let mut step_cycles = vec![];
    let mut previous = 0;
    let mut issue_bound = 0u64;
    while let Some(rec) = reader.next_record(step == steps)? {
        records += 1;
        if records > 200000 || step == steps {
            return Err("extra records or record budget".into());
        }
        match rec {
            Record::Input { source, len } => {
                if !used.insert(source.clone()) {
                    return Err("duplicate source".into());
                }
                let (min, max) = input_len(&source, &h, step)?;
                if len < min || len > max {
                    return Err("input length mismatch".into());
                }
                allocate(
                    &mut allocs,
                    &mut live,
                    &mut peak,
                    &mut cumulative,
                    len,
                    true,
                )?;
                static_checker.bind_input(&source, &[], len)?;
            }
            Record::Alloc { len } => {
                allocate(
                    &mut allocs,
                    &mut live,
                    &mut peak,
                    &mut cumulative,
                    len,
                    false,
                )?;
                static_checker.empty("scratch", len)?;
            }
            Record::Release { tensor } => {
                if allocs.is_empty() || tensor != allocs.len() - 1 {
                    return Err("non-stack release".into());
                }
                let a = allocs.last().unwrap();
                if a.input {
                    return Err("cannot release input".into());
                }
                sched
                    .memory
                    .invalidate_range(a.base as u64, (a.base + 4 * a.len) as u64)?;
                static_checker.release_last(tensor)?;
                allocs.pop();
                live = allocs.last().map_or(0, |a| a.base + 4 * a.len);
            }
            Record::Wave { groups } => {
                let wave = sched.validate_groups(&groups)?;
                static_checker.functional_wave(wave.clone())?;
                let mut wave_issues = vec![0u64; h.hardware.base.sms];
                for g in &groups {
                    wave_issues[g.sm] += g.commands.len() as u64;
                }
                issue_bound += wave_issues.into_iter().max().unwrap_or(0) + 1;
                for stream in &wave {
                    for i in stream {
                        primitives += 1;
                        use Instr::*;
                        work += match i {
                            Mma { m, n, k, .. } | MmaShared { m, n, k, .. } => {
                                (*m as u64) * (*n as u64) * (*k as u64)
                            }
                            Fill { len, .. } | Vector { len, .. } | Reduce { len, .. } => {
                                *len as u64
                            }
                            Load { tensor, global, .. }
                            | Store { tensor, global, .. }
                            | LoadShared { tensor, global, .. }
                            | StoreShared { tensor, global, .. } => {
                                let last = global.end().ok_or("invalid HBM view")?;
                                checked_view(*tensor, 0, last, &allocs)?;
                                global.len() as u64
                            }
                            SharedRead { shared, .. } | SharedWrite { shared, .. } => {
                                shared.len() as u64
                            }
                        };
                    }
                }
                if primitives > 50000000 || work > 200000000000 {
                    return Err("work budget".into());
                }
                let bases: Vec<_> = allocs.iter().map(|a| a.base).collect();
                sched.wave(&groups, &bases)?;
            }
            Record::Commit { outputs } => {
                if !used.contains(&format!("input{step}"))
                    || outputs.keys.len() != h.shape.layers
                    || outputs.values.len() != h.shape.layers
                {
                    return Err("missing input/layer output".into());
                }
                let size = h.batch * if h.decode { 1 } else { h.shape.prefill } * h.shape.d;
                for v in std::iter::once(&outputs.hidden)
                    .chain(&outputs.keys)
                    .chain(&outputs.values)
                {
                    if v.len != size {
                        return Err("output length mismatch".into());
                    }
                    checked_view(v.tensor, v.base, v.len, &allocs)?;
                }
                // Commit does not establish numerical correctness in this mode.
                sched.wave(&[], &allocs.iter().map(|a| a.base).collect::<Vec<_>>())?;
                issue_bound += 1;
                step_cycles.push(sched.stats.cycles - previous);
                previous = sched.stats.cycles;
                step += 1;
            }
        }
        if sched.stats.cycles > 8000000000 {
            return Err("scenario cycle budget".into());
        }
        if static_checker.hbm_usage() != (live, peak) {
            return Err("allocation model differs from official static checker".into());
        }
    }
    if step != steps {
        return Err("missing commits".into());
    }
    // A cheap issue/fence lower bound here; independent Python microstep bounds
    // are computed before MIP proposals. Do not re-expand millions of profiles
    // during timing replay (the immutable scheduler already memoizes them).
    let lower = issue_bound.max(1);
    Ok(
        json!({"report":sched.report(),"header":h,"program_sha256":reader.hash(),
        "hbm_peak_allocated_bytes":peak,"cumulative_allocated_bytes":4*cumulative,
        "step_cycles":step_cycles,"work_units":work,"primitives":primitives,
        "lower_bound_cycles":lower,
        "numerical_correctness":"not_checked",
        "eligibility":"not_established_without_functional_evaluation",
        "checks_omitted":["initialization and numerical execution","reference comparison","host timeout/RSS"]}),
    )
}
fn allocate(
    allocs: &mut Vec<Allocation>,
    live: &mut usize,
    peak: &mut usize,
    cumulative: &mut u64,
    len: usize,
    input: bool,
) -> Result<()> {
    if len == 0 {
        return Err("empty allocation".into());
    }
    *cumulative = cumulative
        .checked_add(len as u64)
        .ok_or("allocation overflow")?;
    if *cumulative > 16u64 << 30 {
        return Err("cumulative allocation budget".into());
    }
    let base = live.next_multiple_of(256);
    *live = base
        .checked_add(len.checked_mul(4).ok_or("allocation overflow")?)
        .ok_or("allocation overflow")?;
    if *live > 2usize << 30 {
        return Err("HBM capacity".into());
    }
    *peak = (*peak).max(*live);
    allocs.push(Allocation { base, len, input });
    Ok(())
}
