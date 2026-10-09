//! Versioned, data-independent batch program format. The evaluator never executes student host code.
use crate::{
    Result,
    baseline::{Fixture, FixtureProfile, Outputs, Policy, Shape},
    explore,
    machine::Machine,
    reference,
    submission::Checkpoint,
    v09_hardware::Hardware,
    v09_schedule::{Group, Report, native_groups},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::BufReader;
use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
pub const CONTRACT: &str = if cfg!(feature = "compact") {
    "phase-two-static-v1"
} else if cfg!(feature = "long-waves") {
    "phase-two-long-waves-static-v1"
} else {
    "vnext-static-v09"
};
fn one() -> usize {
    1
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub shape: Shape,
    pub batch: usize,
    pub decode: bool,
    pub hardware: Hardware,
    pub policy: Policy,
    #[serde(default)]
    pub fixture_profile: FixtureProfile,
    #[serde(default = "one")]
    pub reference_threads: usize,
    #[serde(default = "one")]
    pub parallel_groups: usize,
    #[serde(default)]
    pub async_prefetch: bool,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        self.hardware.validate()?;
        if self.parallel_groups == 0 || self.parallel_groups > self.hardware.resident_groups {
            return Err("invalid parallel_groups".into());
        }
        self.fixture_config().validate()
    }
    fn fixture_config(&self) -> explore::Config {
        explore::Config {
            shape: self.shape,
            batch: self.batch,
            decode: self.decode,
            hardware: crate::hardware::Hardware::default(),
            policy: self.policy,
            fixture_profile: self.fixture_profile,
            reference_threads: self.reference_threads,
        }
    }
    pub fn fixture(&self, seed: u64) -> Result<Fixture> {
        self.validate()?;
        self.fixture_config().fixture(seed)
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub contract: String,
    pub model: String,
    pub hardware: Hardware,
    pub shape: Shape,
    pub batch: usize,
    pub decode: bool,
}
#[derive(Serialize)]
#[cfg_attr(not(feature = "long-waves"), derive(Deserialize))]
#[serde(tag = "record", rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
    Input { source: String, len: usize },
    Alloc { len: usize },
    Release { tensor: usize },
    Wave { groups: Vec<Group> },
    Commit { outputs: Checkpoint },
}
// Large waves must deserialize directly into typed groups. The derived internally
// tagged enum otherwise buffers the whole wave in Serde's generic Content tree.
#[cfg(feature = "long-waves")]
impl<'de> Deserialize<'de> for Record {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Record;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a static program record")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Record, M::Error> {
                use serde::de::Error;
                let (mut tag, mut source, mut len, mut tensor, mut groups, mut outputs) =
                    (None, None, None, None, None, None);
                let mut seen = 0u8;
                while let Some(key) = map.next_key::<String>()? {
                    let bit = match key.as_str() {
                        "record" => 1,
                        "source" => 2,
                        "len" => 4,
                        "tensor" => 8,
                        "groups" => 16,
                        "outputs" => 32,
                        _ => return Err(M::Error::custom("unknown field")),
                    };
                    if seen & bit != 0 {
                        return Err(M::Error::custom("duplicate record field"));
                    }
                    seen |= bit;
                    match bit {
                        1 => tag = Some(map.next_value::<String>()?),
                        2 => source = Some(map.next_value::<String>()?),
                        4 => len = Some(map.next_value::<usize>()?),
                        8 => tensor = Some(map.next_value::<usize>()?),
                        16 => groups = Some(map.next_value::<Vec<Group>>()?),
                        32 => outputs = Some(map.next_value::<Checkpoint>()?),
                        _ => unreachable!(),
                    }
                }
                // Exact masks reject missing fields and fields belonging to another variant.
                match (tag.as_deref(), seen) {
                    (Some("input"), 7) => Ok(Record::Input {
                        source: source.unwrap(),
                        len: len.unwrap(),
                    }),
                    (Some("alloc"), 5) => Ok(Record::Alloc { len: len.unwrap() }),
                    (Some("release"), 9) => Ok(Record::Release {
                        tensor: tensor.unwrap(),
                    }),
                    (Some("wave"), 17) => Ok(Record::Wave {
                        groups: groups.unwrap(),
                    }),
                    (Some("commit"), 33) => Ok(Record::Commit {
                        outputs: outputs.unwrap(),
                    }),
                    _ => Err(M::Error::custom(
                        "unknown record or missing/unexpected fields",
                    )),
                }
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

struct ConvertWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    hardware: Hardware,
    asynchronous: bool,
}
impl Write for ConvertWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for &b in bytes {
            if b == b'\n' {
                let old: crate::submission::Record = serde_json::from_slice(&self.buffer)?;
                let r = match old {
                    crate::submission::Record::Input { source, len } => {
                        Record::Input { source, len }
                    }
                    crate::submission::Record::Alloc { len } => Record::Alloc { len },
                    crate::submission::Record::Release { tensor } => Record::Release { tensor },
                    crate::submission::Record::Commit { outputs } => Record::Commit { outputs },
                    crate::submission::Record::Wave { streams } => Record::Wave {
                        groups: native_groups(&streams, &self.hardware, self.asynchronous)
                            .map_err(std::io::Error::other)?,
                    },
                };
                serde_json::to_writer(&mut self.writer, &r)?;
                self.writer.write_all(b"\n")?;
                self.buffer.clear();
            } else {
                self.buffer.push(b);
                if self.buffer.len() as u64 > crate::submission::MAX_LINE_BYTES {
                    return Err(std::io::Error::other("record too large"));
                }
            }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if !self.buffer.is_empty() {
            return Err(std::io::Error::other("incomplete record"));
        }
        self.writer.flush()
    }
}
pub fn export(c: &Config, path: &Path) -> Result<()> {
    c.validate()?;
    let mut writer = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?,
    );
    let header = Header {
        contract: CONTRACT.into(),
        model: crate::MODEL.into(),
        hardware: c.hardware.clone(),
        shape: c.shape,
        batch: c.batch,
        decode: c.decode,
    };
    serde_json::to_writer(&mut writer, &header).map_err(|e| e.to_string())?;
    writer.write_all(b"\n").map_err(|e| e.to_string())?;
    let fixture = c.fixture(0)?;
    let mut m = Machine::new_concurrent(
        c.hardware.clone(),
        false,
        c.parallel_groups,
        c.async_prefetch,
    )?;
    m.record_to(Box::new(ConvertWriter {
        writer,
        buffer: vec![],
        hardware: c.hardware.clone(),
        asynchronous: c.async_prefetch,
    }));
    explore::run_batched(&mut m, &fixture, c.decode, c.policy, c.batch)?;
    m.finish_recording()
}
#[derive(Serialize)]
pub struct Evaluation {
    pub header: Header,
    pub comparison: reference::Comparison,
    pub report: Report,
    pub hbm_peak_allocated_bytes: usize,
    pub program_sha256: String,
    pub output_sha256: String,
    pub step_cycles: Vec<u64>,
    pub work_units: u64,
}
fn output_hash(o: &Outputs) -> String {
    let mut h = Sha256::new();
    for v in o.hidden.iter().chain(&o.keys).chain(&o.values) {
        h.update((v.len() as u64).to_le_bytes());
        for x in v {
            h.update(x.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}
pub fn check(c: &Config, seed: u64) -> Result<serde_json::Value> {
    let f = c.fixture(seed)?;
    let start = std::time::Instant::now();
    let expected = reference::run_batched(&f, c.decode, c.batch, c.reference_threads);
    let reference_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let mut m = Machine::new_concurrent(
        c.hardware.clone(),
        true,
        c.parallel_groups,
        c.async_prefetch,
    )?;
    let actual = explore::run_batched(&mut m, &f, c.decode, c.policy, c.batch)?;
    let comparison = reference::compare(&actual, &expected)?;
    let report = m.concurrent_report()?;
    Ok(
        serde_json::json!({"comparison":comparison,"report":report,"hbm_peak_allocated_bytes":m.hbm_usage().1,"output_sha256":output_hash(&actual),"reference_seconds":reference_seconds,"simulation_seconds":start.elapsed().as_secs_f64()}),
    )
}
pub fn evaluate(path: &Path, seed: u64, profile: FixtureProfile) -> Result<Evaluation> {
    #[cfg(feature = "compact")]
    let mut compact = crate::compact::Reader::open(path)?;
    #[cfg(feature = "compact")]
    let header = compact.header.clone();
    #[cfg(not(feature = "compact"))]
    let (mut reader, mut buf, mut total, mut hash, header) = {
        let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut buf = vec![];
        let mut total = 0;
        let mut hash = Sha256::new();
        if !crate::submission::line(&mut reader, &mut buf, &mut total)? {
            return Err("empty program".into());
        }
        hash.update(&buf);
        let header: Header = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
        (reader, buf, total, hash, header)
    };
    if header.contract != CONTRACT || header.model != crate::MODEL {
        return Err("wrong model/contract".into());
    }
    let c = Config {
        shape: header.shape,
        batch: header.batch,
        decode: header.decode,
        hardware: header.hardware.clone(),
        policy: Policy::square(16),
        fixture_profile: profile,
        reference_threads: 1,
        parallel_groups: 1,
        async_prefetch: false,
    };
    let f = c.fixture(seed)?;
    let expected = reference::run_batched(&f, c.decode, c.batch, 1);
    let mut m = Machine::new_concurrent(c.hardware.clone(), true, 1, false)?;
    let mut used = HashSet::new();
    let mut protected = HashSet::new();
    let (mut step, mut records, mut instructions, mut work, mut allocated) =
        (0usize, 0usize, 0u64, 0u64, 0u64);
    let steps = if c.decode { c.shape.steps } else { 1 };
    let rows = c.batch * if c.decode { 1 } else { c.shape.prefill };
    let size = rows * c.shape.d;
    let mut actual = Outputs {
        hidden: vec![],
        keys: vec![vec![]; c.shape.layers],
        values: vec![vec![]; c.shape.layers],
    };
    let mut step_cycles = vec![];
    let mut previous = 0;
    while let Some(record) = {
        #[cfg(feature = "compact")]
        {
            compact.next_record(step == steps)?
        }
        #[cfg(not(feature = "compact"))]
        {
            if crate::submission::line(&mut reader, &mut buf, &mut total)? {
                hash.update(&buf);
                Some(serde_json::from_slice::<Record>(&buf).map_err(|e| e.to_string())?)
            } else {
                None
            }
        }
    } {
        records += 1;
        if records > 200_000 || step == steps {
            return Err("extra records or records budget".into());
        }
        match record {
            Record::Input { source, len } => {
                if !used.insert(source.clone()) {
                    return Err("input may be bound only once".into());
                }
                let data = crate::submission::source(&f, &source, c.decode, step)?;
                let max = if source.ends_with(".past_k") || source.ends_with(".past_v") {
                    (c.shape.history + c.shape.steps) * c.batch * c.shape.d
                } else {
                    data.len()
                };
                if len < data.len() || len > max {
                    return Err("input length mismatch".into());
                }
                allocated = allocated
                    .checked_add(len as u64)
                    .ok_or("allocation overflow")?;
                let limit = if cfg!(feature = "compact") {
                    16u64 << 30
                } else {
                    2u64 << 30
                };
                if allocated > limit {
                    return Err("cumulative allocation budget exceeded".into());
                }
                let id = m.bind_input(&source, data, len)?;
                protected.insert(id);
            }
            Record::Alloc { len } => {
                if len == 0 {
                    return Err("empty allocation".into());
                }
                allocated = allocated
                    .checked_add(len as u64)
                    .ok_or("allocation overflow")?;
                let limit = if cfg!(feature = "compact") {
                    16u64 << 30
                } else {
                    2u64 << 30
                };
                if allocated > limit {
                    return Err("cumulative allocation budget exceeded".into());
                }
                m.empty("scratch", len)?;
            }
            Record::Release { tensor } => {
                if protected.contains(&tensor) {
                    return Err("cannot release bound input".into());
                }
                m.release_last(tensor)?;
            }
            Record::Wave { groups } => {
                let wave = m.concurrent.as_ref().unwrap().validate_groups(&groups)?;
                let (count, cost) = crate::submission::work(&wave)?;
                instructions = instructions
                    .checked_add(count)
                    .ok_or("instruction overflow")?;
                work = work.checked_add(cost).ok_or("work overflow")?;
                if instructions
                    > if cfg!(feature = "compact") {
                        50_000_000
                    } else {
                        20_000_000
                    }
                    || work > 200_000_000_000
                {
                    return Err("scenario work budget exceeded".into());
                }
                m.concurrent_wave(groups)?;
            }
            Record::Commit { outputs } => {
                if outputs.keys.len() != c.shape.layers
                    || outputs.values.len() != c.shape.layers
                    || !used.contains(&format!("input{step}"))
                {
                    return Err("missing input/layer output".into());
                }
                let mut part = Outputs {
                    hidden: vec![crate::submission::gather(&m, &outputs.hidden, size)?],
                    keys: vec![],
                    values: vec![],
                };
                for (a, b) in outputs.keys.iter().zip(&outputs.values) {
                    part.keys.push(crate::submission::gather(&m, a, size)?);
                    part.values.push(crate::submission::gather(&m, b, size)?);
                }
                let refpart = Outputs {
                    hidden: vec![expected.hidden[step].clone()],
                    keys: expected
                        .keys
                        .iter()
                        .map(|v| v[step * size..(step + 1) * size].to_vec())
                        .collect(),
                    values: expected
                        .values
                        .iter()
                        .map(|v| v[step * size..(step + 1) * size].to_vec())
                        .collect(),
                };
                reference::compare(&part, &refpart)?;
                actual.hidden.extend(part.hidden);
                for (a, v) in actual.keys.iter_mut().zip(part.keys) {
                    a.extend(v);
                }
                for (a, v) in actual.values.iter_mut().zip(part.values) {
                    a.extend(v);
                }
                m.concurrent_wave(vec![])?;
                let cycles = m.concurrent.as_ref().unwrap().stats.cycles;
                step_cycles.push(cycles - previous);
                previous = cycles;
                step += 1;
            }
        }
        if allocated
            > if cfg!(feature = "compact") {
                16u64 * 1024 * 1024 * 1024
            } else {
                2u64 * 1024 * 1024 * 1024
            }
        {
            return Err("cumulative allocation budget exceeded".into());
        }
        if m.concurrent.as_ref().unwrap().stats.cycles > 8_000_000_000 {
            return Err("scenario cycle budget exceeded".into());
        }
    }
    if step != steps {
        return Err("missing commits".into());
    }
    let report = m.concurrent_report()?;
    if !report.power_pass {
        return Err("power constraint failed".into());
    }
    Ok(Evaluation {
        header,
        comparison: reference::compare(&actual, &expected)?,
        report,
        hbm_peak_allocated_bytes: m.hbm_usage().1,
        program_sha256: {
            #[cfg(feature = "compact")]
            {
                compact.hash()
            }
            #[cfg(not(feature = "compact"))]
            {
                format!("{:x}", hash.finalize())
            }
        },
        output_sha256: output_hash(&actual),
        step_cycles,
        work_units: work,
    })
}
pub fn known_shape(name: &str) -> Option<(Shape, usize, bool)> {
    let mut s = Shape {
        layers: 2,
        d: 1536,
        heads: 12,
        f: 6144,
        prefill: 1,
        history: 4096,
        steps: 4,
    };
    Some(match name {
        "w-p" => {
            s.prefill = 256;
            s.history = 1;
            s.steps = 1;
            (s, 1, false)
        }
        "a-p" => {
            s.d = 512;
            s.heads = 8;
            s.f = 2048;
            s.prefill = 2048;
            s.history = 1;
            s.steps = 1;
            (s, 1, false)
        }
        "w-d1" => (s, 1, true),
        "w-d16" => {
            s.history = 256;
            (s, 16, true)
        }
        "w-d4l" => (s, 4, true),
        _ => return None,
    })
}

pub const CASES: [&str; 5] = ["w-p", "a-p", "w-d1", "w-d16", "w-d4l"];
// Populated only after the reference design is fully checked under final costs.
pub const REFERENCE_CYCLES: [u64; 5] =
    [187_689_274, 374_856_358, 27_068_433, 42_358_913, 70_439_292];
#[derive(Serialize)]
pub struct Grade {
    pub contract: &'static str,
    pub model: &'static str,
    pub source_sha256: String,
    pub score: f64,
    pub evaluations: Vec<Evaluation>,
}
pub fn grade(directory: &Path, seed: u64, profile: FixtureProfile) -> Result<Grade> {
    let mut hardware = None;
    let mut paths = vec![];
    // Admission checks every header before doing expensive numerical work.
    for name in CASES {
        let path = directory.join(format!("{name}.jsonl"));
        let mut r = BufReader::new(File::open(&path).map_err(|e| e.to_string())?);
        let (mut buf, mut total) = (vec![], 0);
        if !crate::submission::line(&mut r, &mut buf, &mut total)? {
            return Err("empty program".into());
        }
        let h: Header = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
        let (shape, batch, decode) = known_shape(name).unwrap();
        if serde_json::to_value(h.shape).unwrap() != serde_json::to_value(shape).unwrap()
            || h.batch != batch
            || h.decode != decode
        {
            return Err(format!("fixed workload mismatch: {name}"));
        }
        if h.contract != CONTRACT || h.model != crate::MODEL {
            return Err("wrong contract/model".into());
        }
        h.hardware.validate()?;
        let value = serde_json::to_value(&h.hardware).unwrap();
        if let Some(previous) = &hardware {
            if previous != &value {
                return Err("all five programs must use identical hardware".into());
            }
        } else {
            hardware = Some(value);
        }
        paths.push(path);
    }
    if REFERENCE_CYCLES.contains(&0) {
        return Err("reference scores not yet frozen".into());
    }
    let mut evaluations = vec![];
    let mut log_score = 0.;
    for (i, path) in paths.iter().enumerate() {
        let e = evaluate(path, seed, profile)?;
        let (shape, batch, decode) = known_shape(CASES[i]).unwrap();
        if serde_json::to_value(e.header.shape).unwrap() != serde_json::to_value(shape).unwrap()
            || e.header.batch != batch
            || e.header.decode != decode
            || Some(serde_json::to_value(&e.header.hardware).unwrap()) != hardware
        {
            return Err("program header changed after admission".into());
        }
        let cycles = e.report.stats.cycles;
        if cycles == 0 {
            return Err("zero cycles".into());
        }
        let weight = if i < 2 { 0.25 } else { 1. / 6. };
        log_score += weight * (REFERENCE_CYCLES[i] as f64 / cycles as f64).ln();
        evaluations.push(e);
    }
    Ok(Grade {
        contract: CONTRACT,
        model: crate::MODEL,
        source_sha256: crate::source::hash(),
        score: 1000. * log_score.exp(),
        evaluations,
    })
}
