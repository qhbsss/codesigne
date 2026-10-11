//! Data-independent, bounded, streaming submission contract.
//! Student code is never loaded into the evaluator process.
use crate::{
    Result,
    baseline::{self, Fixture, FixtureProfile, Outputs, Shape},
    hardware::Hardware,
    isa::{Instr, Wave},
    machine::{Machine, Report},
    reference::{self, Comparison},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

pub const CONTRACT: &str = "vnext-static-sh-v1";
pub const MAX_LINE_BYTES: u64 = if cfg!(feature = "compact") {
    16 * 1024 * 1024
} else if cfg!(feature = "long-waves") {
    64 * 1024 * 1024
} else {
    8 * 1024 * 1024
};
pub const MAX_FILE_BYTES: u64 = if cfg!(feature = "compact") {
    128 * 1024 * 1024
} else if cfg!(feature = "long-waves") {
    4 * 1024 * 1024 * 1024
} else {
    1024 * 1024 * 1024
};
pub const MAX_RECORDS: usize = 200_000;
pub const MAX_INSTRUCTIONS: u64 = 10_000_000;
pub const MAX_WORK: u64 = 50_000_000_000;
pub const MAX_CYCLES: u64 = 400_000_000;
pub const MAX_STEP_CYCLES: u64 = 100_000_000;
// Frozen ordinary starter, never a claim of optimum. Exact values verified at release.
pub const REFERENCE_PREFILL: u64 = 14_227_707;
pub const REFERENCE_DECODE: u64 = 9_938_724;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub contract: String,
    pub model: String,
    pub hardware: Hardware,
    pub shape: Shape,
    pub decode: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputView {
    pub tensor: usize,
    pub base: usize,
    pub len: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub hidden: OutputView,
    pub keys: Vec<OutputView>,
    pub values: Vec<OutputView>,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "record", rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
    Input { source: String, len: usize },
    Alloc { len: usize },
    Release { tensor: usize },
    Wave { streams: Wave },
    Commit { outputs: Checkpoint },
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    pub shape: Shape,
    pub hardware: Hardware,
    pub tile: usize,
    pub seed: u64,
    #[serde(default)]
    pub fixture_profile: FixtureProfile,
    #[serde(default)]
    pub prefill_policy: Option<baseline::Policy>,
    #[serde(default)]
    pub decode_policy: Option<baseline::Policy>,
}
fn equal_json(a: &impl Serialize, b: &impl Serialize) -> bool {
    serde_json::to_value(a).ok() == serde_json::to_value(b).ok()
}
impl Header {
    pub fn validate(&self) -> Result<()> {
        if cfg!(feature = "explore") {
            return Err("experimental model has no frozen static submission contract".into());
        }
        if self.contract != CONTRACT || self.model != crate::MODEL {
            return Err("unsupported release contract/model".into());
        }
        self.shape.validate()?;
        self.hardware.validate()?;
        if self.hardware.sh_tc_bw == 128 {
            return Err("128 B/cycle SH interface excluded from release: dominated by 64".into());
        }
        Ok(())
    }
    pub fn is_main(&self) -> bool {
        equal_json(&self.shape, &Shape::main())
    }
}
/// Bounds are checked before deserializing a line, including its terminating newline.
pub(crate) fn line(reader: &mut impl BufRead, buf: &mut Vec<u8>, total: &mut u64) -> Result<bool> {
    buf.clear();
    reader
        .take(MAX_LINE_BYTES + 1)
        .read_until(b'\n', buf)
        .map_err(|e| e.to_string())?;
    if buf.len() as u64 > MAX_LINE_BYTES {
        return Err("record exceeds line-byte admission limit".into());
    }
    *total += buf.len() as u64;
    if *total > MAX_FILE_BYTES {
        return Err("program exceeds file-byte admission limit".into());
    }
    Ok(!buf.is_empty())
}
fn parse<T: for<'a> Deserialize<'a>>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}
pub fn read_header(path: &Path) -> Result<Header> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut buf = vec![];
    if !line(&mut reader, &mut buf, &mut 0)? {
        return Err("empty program".into());
    }
    let h: Header = parse(&buf)?;
    h.validate()?;
    Ok(h)
}
pub fn pair_headers(dir: &Path, main_only: bool) -> Result<(Header, Header)> {
    let p = read_header(&dir.join("prefill.jsonl"))?;
    let d = read_header(&dir.join("decode.jsonl"))?;
    if p.decode
        || !d.decode
        || !equal_json(&p.hardware, &d.hardware)
        || !equal_json(&p.shape, &d.shape)
    {
        return Err(
            "P/D must use one identical hardware and shape, with correct scenario flags".into(),
        );
    }
    if main_only && !p.is_main() {
        return Err("grade requires the frozen M shape".into());
    }
    Ok((p, d))
}
pub fn export(config: &ExportConfig, dir: &Path) -> Result<()> {
    config.shape.validate()?;
    config.hardware.validate()?;
    if ![8, 16, 32].contains(&config.tile) {
        return Err("invalid tile".into());
    }
    let fixture = Fixture::new(config.shape, 0)?;
    std::fs::create_dir(dir).map_err(|e| e.to_string())?;
    for decode in [false, true] {
        let header = Header {
            contract: CONTRACT.into(),
            model: crate::MODEL.into(),
            shape: config.shape,
            hardware: config.hardware.clone(),
            decode,
        };
        header.validate()?;
        let policy = (if decode {
            config.decode_policy
        } else {
            config.prefill_policy
        })
        .unwrap_or_else(|| baseline::Policy::square(config.tile));
        policy.validate()?;
        let file = File::create(dir.join(if decode {
            "decode.jsonl"
        } else {
            "prefill.jsonl"
        }))
        .map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &header).map_err(|e| e.to_string())?;
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
        let mut machine = Machine::new(config.hardware.clone(), false)?;
        machine.record_to(Box::new(writer));
        baseline::run_policy(&mut machine, &fixture, decode, policy)?;
        machine.finish_recording()?;
    }
    Ok(())
}
pub(crate) fn source<'a>(
    f: &'a Fixture,
    name: &str,
    decode: bool,
    step: usize,
) -> Result<&'a [f32]> {
    // Canonical spellings prevent aliasing an input under multiple source names.
    if let Some(index) = name.strip_prefix("input") {
        let i: usize = index.parse().map_err(|_| "invalid input source")?;
        if name != format!("input{i}") || i != step || (!decode && i != 0) {
            return Err("future/past input is not released".into());
        }
        return if decode {
            f.inputs
                .get(i)
                .map(Vec::as_slice)
                .ok_or("input index".into())
        } else {
            Ok(&f.prompt)
        };
    }
    let (prefix, field) = name.split_once('.').ok_or("invalid source")?;
    let i: usize = prefix
        .strip_prefix("layer")
        .ok_or("invalid layer")?
        .parse()
        .map_err(|_| "invalid layer")?;
    if prefix != format!("layer{i}") {
        return Err("noncanonical layer source".into());
    }
    let l = f.layers.get(i).ok_or("layer index")?;
    Ok(match field {
        "qkv" => &l.qkv,
        "qb" => &l.qb,
        "out" => &l.out,
        "ob" => &l.ob,
        "up" => &l.up,
        "ub" => &l.ub,
        "down" => &l.down,
        "db" => &l.db,
        "g1" => &l.g1,
        "b1" => &l.b1,
        "g2" => &l.g2,
        "b2" => &l.b2,
        "past_k" if decode => &l.past_k,
        "past_v" if decode => &l.past_v,
        _ => return Err("unknown/unavailable fixture source".into()),
    })
}
pub(crate) fn gather(machine: &Machine, view: &OutputView, len: usize) -> Result<Vec<f32>> {
    if view.len != len {
        return Err("output length mismatch".into());
    }
    let tensor = machine
        .tensors
        .get(view.tensor)
        .ok_or("unknown output tensor")?;
    let end = view.base.checked_add(view.len).ok_or("output overflow")?;
    Ok(tensor
        .data
        .get(view.base..end)
        .ok_or("output out of bounds")?
        .to_vec())
}
/// Checked host-work proxy, separate from architectural timing and score.
pub(crate) fn work(wave: &Wave) -> Result<(u64, u64)> {
    let mut instructions = 0u64;
    let mut work = 0u64;
    for i in wave.iter().flatten() {
        instructions += 1;
        let n = match i {
            Instr::Mma { m, n, k, .. } | Instr::MmaShared { m, n, k, .. } => (*m as u64)
                .checked_mul(*n as u64)
                .and_then(|v| v.checked_mul(*k as u64)),
            Instr::Load { global, .. }
            | Instr::Store { global, .. }
            | Instr::LoadShared { global, .. }
            | Instr::StoreShared { global, .. } => {
                (global.rows as u64).checked_mul(global.cols as u64)
            }
            Instr::SharedRead { shared, .. } | Instr::SharedWrite { shared, .. } => {
                (shared.rows as u64).checked_mul(shared.cols as u64)
            }
            Instr::Fill { len, .. } | Instr::Vector { len, .. } | Instr::Reduce { len, .. } => {
                Some(*len as u64)
            }
        }
        .ok_or("work overflow")?;
        work = work.checked_add(n).ok_or("work overflow")?;
    }
    Ok((instructions, work))
}
#[derive(Serialize)]
pub struct Evaluation {
    pub header: Header,
    pub report: Report,
    pub comparison: Comparison,
    pub step_cycles: Vec<u64>,
    pub program_sha256: String,
    pub output_sha256: String,
    pub records: usize,
    pub work_units: u64,
}
pub fn evaluate(path: &Path, fixture: &Fixture) -> Result<Evaluation> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut buf = vec![];
    let mut total = 0;
    let mut sha = Sha256::new();
    if !line(&mut reader, &mut buf, &mut total)? {
        return Err("empty program".into());
    }
    sha.update(&buf);
    let header: Header = parse(&buf)?;
    header.validate()?;
    if !equal_json(&header.shape, &fixture.shape) {
        return Err("fixture shape mismatch".into());
    }
    let expected = reference::run(fixture, header.decode);
    let mut machine = Machine::new(header.hardware.clone(), true)?;
    let mut actual = Outputs {
        hidden: vec![],
        keys: vec![vec![]; fixture.shape.layers],
        values: vec![vec![]; fixture.shape.layers],
    };
    let mut used = HashSet::new();
    let mut records = 0;
    let mut instructions = 0;
    let mut work_units = 0;
    let mut allocated_words = 0u64;
    let mut step = 0;
    let steps = if header.decode {
        fixture.shape.steps
    } else {
        1
    };
    let rows = if header.decode {
        1
    } else {
        fixture.shape.prefill
    };
    let size = rows * fixture.shape.d;
    let mut step_cycles = vec![];
    let mut previous_cycle = 0;
    while line(&mut reader, &mut buf, &mut total)? {
        sha.update(&buf);
        records += 1;
        if records > MAX_RECORDS {
            return Err("record budget exceeded".into());
        }
        if step == steps {
            return Err("records after final commit".into());
        }
        let record: Record = parse(&buf)?;
        if let Record::Alloc { len } | Record::Input { len, .. } = &record {
            if *len == 0 || machine.tensors.len() >= 65536 {
                return Err("invalid allocation/tensor count".into());
            }
            allocated_words = allocated_words
                .checked_add(*len as u64)
                .ok_or("allocation overflow")?;
            if allocated_words > 2 * 1024 * 1024 * 1024 {
                return Err("cumulative allocation exceeds 8 GiB".into());
            }
        }
        match record {
            Record::Input { source: name, len } => {
                if !used.insert(name.clone()) {
                    return Err("input source can only be bound once".into());
                }
                let data = source(fixture, &name, header.decode, step)?;
                // Only KV inputs allow an uninitialized appended output region.
                let max = if name.ends_with(".past_k") || name.ends_with(".past_v") {
                    (fixture.shape.history + fixture.shape.steps) * fixture.shape.d
                } else {
                    data.len()
                };
                if len < data.len() || len > max {
                    return Err("input allocation length mismatch".into());
                }
                machine.bind_input(&name, data, len)?;
            }
            Record::Alloc { len } => {
                machine.empty("scratch", len)?;
            }
            Record::Release { tensor } => machine.release_last(tensor)?,
            Record::Wave { streams } => {
                let (count, cost) = work(&streams)?;
                instructions += count;
                work_units += cost;
                if instructions > MAX_INSTRUCTIONS || work_units > MAX_WORK {
                    return Err("scenario instruction/work budget exceeded".into());
                }
                machine.wave(streams)?;
            }
            Record::Commit { outputs } => {
                if outputs.keys.len() != fixture.shape.layers
                    || outputs.values.len() != fixture.shape.layers
                {
                    return Err("commit requires every layer KV".into());
                }
                if !used.contains(&format!("input{step}")) {
                    return Err("commit without current input".into());
                }
                let mut part = Outputs {
                    hidden: vec![gather(&machine, &outputs.hidden, size)?],
                    keys: vec![],
                    values: vec![],
                };
                for (a, b) in outputs.keys.iter().zip(&outputs.values) {
                    part.keys.push(gather(&machine, a, size)?);
                    part.values.push(gather(&machine, b, size)?);
                }
                let reference_part = Outputs {
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
                reference::compare(&part, &reference_part)?;
                actual.hidden.extend(part.hidden);
                for (dst, src) in actual.keys.iter_mut().zip(part.keys) {
                    dst.extend(src);
                }
                for (dst, src) in actual.values.iter_mut().zip(part.values) {
                    dst.extend(src);
                }
                machine.checkpoint(outputs)?;
                let cycles = machine.timing.stats.cycles;
                if header.decode && cycles - previous_cycle > MAX_STEP_CYCLES {
                    return Err("decode step cycle limit exceeded".into());
                }
                step_cycles.push(cycles - previous_cycle);
                previous_cycle = cycles;
                step += 1;
            }
        }
        if machine.timing.stats.cycles > MAX_CYCLES {
            return Err("scenario cycle limit exceeded".into());
        }
    }
    if step != steps {
        return Err("missing output commits".into());
    }
    let report = machine.report();
    if !report.power_pass || !report.execution_valid {
        return Err("power/execution qualification failed".into());
    }
    let comparison = reference::compare(&actual, &expected)?;
    let mut output = Sha256::new();
    for v in actual
        .hidden
        .iter()
        .chain(&actual.keys)
        .chain(&actual.values)
    {
        output.update((v.len() as u64).to_le_bytes());
        for x in v {
            output.update(x.to_le_bytes());
        }
    }
    Ok(Evaluation {
        header,
        report,
        comparison,
        step_cycles,
        program_sha256: format!("{:x}", sha.finalize()),
        output_sha256: format!("{:x}", output.finalize()),
        records,
        work_units,
    })
}

pub fn score(prefill: u64, decode: u64) -> Result<f64> {
    if prefill == 0 || decode == 0 {
        return Err("zero-cycle score".into());
    }
    Ok(1000.
        * ((REFERENCE_PREFILL as f64 / prefill as f64) * (REFERENCE_DECODE as f64 / decode as f64))
            .sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_reuse_wave_has_no_extra_host_work_gate() {
        // Valid shared-MMA dimensions and 1024 instructions can exceed the
        // rejected 250M host threshold while fitting the public ISA envelope.
        let streams = vec![vec![
            Instr::MmaShared {
                a: 0,
                b: crate::isa::View::contiguous(0, 192, 64),
                c: 64 * 192,
                m: 64,
                n: 64,
                k: 192,
            };
            1024
        ]];
        assert_eq!(work(&streams).unwrap(), (1024, 805_306_368));
    }
    #[test]
    fn overflowing_work_is_rejected_before_execution() {
        let streams = vec![vec![Instr::Mma {
            a: 0,
            b: 0,
            c: 0,
            m: usize::MAX,
            n: usize::MAX,
            k: usize::MAX,
        }]];
        assert!(work(&streams).is_err());
    }
}
