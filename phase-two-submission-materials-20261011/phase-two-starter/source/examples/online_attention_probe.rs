//! Paid online-softmax attention probe. No fused/unpriced attention primitive.
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    isa::{Arg, Instr, Op, View},
    machine::Machine,
    v09_hardware::Hardware,
    v09_schedule::{Command, Group},
};
fn reg(base: usize) -> Arg {
    Arg::Reg { base, stride: 1 }
}
fn scalar(base: usize) -> Arg {
    Arg::Reg { base, stride: 0 }
}
fn imm(value: f32) -> Arg {
    Arg::Imm { value }
}
fn emit(c: &mut Vec<Command>, i: Instr) {
    c.push(Command::Run { instruction: i });
}
fn vecop(c: &mut Vec<Command>, kind: Op, dst: usize, len: usize, a: Arg, b: Arg) {
    emit(
        c,
        Instr::Vector {
            kind,
            dst,
            len,
            a,
            b,
        },
    );
}
fn fill(c: &mut Vec<Command>, dst: usize, len: usize, value: f32) {
    emit(c, Instr::Fill { dst, len, value });
}
fn load(
    c: &mut Vec<Command>,
    tensor: usize,
    global: View,
    local: View,
    asynchronous: bool,
    token: &mut u32,
) -> Option<u32> {
    let instruction = Instr::Load {
        tensor,
        global,
        local,
    };
    if asynchronous {
        *token += 1;
        c.push(Command::Async {
            token: *token,
            instruction,
        });
        Some(*token)
    } else {
        emit(c, instruction);
        None
    }
}
fn view(base: usize, rows: usize, cols: usize, rs: usize, cs: usize) -> View {
    View {
        base,
        rows,
        cols,
        row_stride: rs,
        col_stride: cs,
    }
}
#[allow(clippy::too_many_arguments)]
fn group(
    start: usize,
    rows: usize,
    context: usize,
    hd: usize,
    block: usize,
    causal: bool,
    sm: usize,
    asynchronous: bool,
) -> Group {
    // Q, O, score/probability, two alternating K^T/V buffers, row m/l and scratch.
    let q = 0;
    let o = q + rows * hd;
    let score = o + rows * hd;
    let kb = score + rows * block;
    let vb = kb + 2 * hd * block;
    let max = vb + 2 * block * hd;
    let sum = max + rows;
    let tmp = sum + rows;
    let scratch = tmp + 8;
    let words = scratch + block.max(2);
    let mut c = vec![];
    let mut token = 0;
    load(
        &mut c,
        0,
        view(start * hd, rows, hd, hd, 1),
        View::contiguous(q, rows, hd),
        false,
        &mut token,
    );
    fill(&mut c, o, rows * hd, 0.);
    fill(&mut c, max, rows, -1e30);
    fill(&mut c, sum, rows, 0.);
    let end = if causal {
        (start + rows).min(context)
    } else {
        context
    };
    let mut pending = vec![];
    for (iteration, begin) in (0..end).step_by(block).enumerate() {
        let n = (end - begin).min(block);
        let slot = iteration % 2;
        let k = kb + slot * hd * block;
        let v = vb + slot * hd * block;
        if iteration == 0 || !asynchronous {
            for t in [
                load(
                    &mut c,
                    1,
                    view(begin * hd, hd, n, 1, hd),
                    View::contiguous(k, hd, n),
                    asynchronous,
                    &mut token,
                ),
                load(
                    &mut c,
                    2,
                    view(begin * hd, n, hd, hd, 1),
                    View::contiguous(v, n, hd),
                    asynchronous,
                    &mut token,
                ),
            ]
            .into_iter()
            .flatten()
            {
                pending.push(t);
            }
        }
        if !pending.is_empty() {
            c.push(Command::Wait {
                tokens: std::mem::take(&mut pending),
            });
        }
        // Prefetch next pair into a disjoint ping-pong buffer while computing this block.
        if asynchronous && begin + n < end {
            let next = begin + n;
            let nn = (end - next).min(block);
            let slot = 1 - slot;
            for t in [
                load(
                    &mut c,
                    1,
                    view(next * hd, hd, nn, 1, hd),
                    View::contiguous(kb + slot * hd * block, hd, nn),
                    true,
                    &mut token,
                ),
                load(
                    &mut c,
                    2,
                    view(next * hd, nn, hd, hd, 1),
                    View::contiguous(vb + slot * hd * block, nn, hd),
                    true,
                    &mut token,
                ),
            ]
            .into_iter()
            .flatten()
            {
                pending.push(t);
            }
        }
        fill(&mut c, score, rows * n, 0.);
        emit(
            &mut c,
            Instr::Mma {
                a: q,
                b: k,
                c: score,
                m: rows,
                n,
                k: hd,
            },
        );
        vecop(
            &mut c,
            Op::Mul,
            score,
            rows * n,
            reg(score),
            imm(1. / (hd as f32).sqrt()),
        );
        for r in 0..rows {
            let p = score + r * n;
            let valid = if causal {
                (start + r + 1).saturating_sub(begin).min(n)
            } else {
                n
            };
            if valid < n {
                fill(&mut c, p + valid, n - valid, -1e30);
            }
            emit(
                &mut c,
                Instr::Reduce {
                    scratch,
                    dst: tmp,
                    src: p,
                    len: n,
                    max: true,
                },
            );
            vecop(&mut c, Op::Add, tmp + 1, 1, scalar(max + r), imm(0.));
            emit(
                &mut c,
                Instr::Reduce {
                    scratch,
                    dst: tmp + 2,
                    src: tmp,
                    len: 2,
                    max: true,
                },
            );
            vecop(
                &mut c,
                Op::Sub,
                tmp + 3,
                1,
                scalar(max + r),
                scalar(tmp + 2),
            );
            vecop(&mut c, Op::Exp, tmp + 3, 1, scalar(tmp + 3), imm(0.));
            vecop(&mut c, Op::Sub, p, n, reg(p), scalar(tmp + 2));
            vecop(&mut c, Op::Exp, p, n, reg(p), imm(0.));
            emit(
                &mut c,
                Instr::Reduce {
                    scratch,
                    dst: tmp + 4,
                    src: p,
                    len: n,
                    max: false,
                },
            );
            vecop(
                &mut c,
                Op::Mul,
                sum + r,
                1,
                scalar(sum + r),
                scalar(tmp + 3),
            );
            vecop(
                &mut c,
                Op::Add,
                sum + r,
                1,
                scalar(sum + r),
                scalar(tmp + 4),
            );
            vecop(
                &mut c,
                Op::Mul,
                o + r * hd,
                hd,
                reg(o + r * hd),
                scalar(tmp + 3),
            );
            vecop(&mut c, Op::Add, max + r, 1, scalar(tmp + 2), imm(0.));
        }
        // hd may exceed the software N limit; output slices use paid strided V packing.
        assert!(hd <= 64);
        emit(
            &mut c,
            Instr::Mma {
                a: score,
                b: v,
                c: o,
                m: rows,
                n: hd,
                k: n,
            },
        );
    }
    for r in 0..rows {
        vecop(
            &mut c,
            Op::Div,
            o + r * hd,
            hd,
            reg(o + r * hd),
            scalar(sum + r),
        );
    }
    emit(
        &mut c,
        Instr::Store {
            tensor: 3,
            global: view(start * hd, rows, hd, hd, 1),
            local: View::contiguous(o, rows, hd),
        },
    );
    Group {
        sm,
        rf_kib: words.div_ceil(256),
        sh_kib: 0,
        commands: c,
    }
}
#[derive(serde::Serialize)]
struct WaveRecord<'a> {
    record: &'static str,
    groups: &'a [Group],
}
fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + salt * 101) % 1009) as f32 / 1009. - 0.5) * 3.)
        .collect()
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("CONFIG CONTEXT TILE GROUPS ASYNC NEW_OUTPUT_JSON".into());
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let h: Hardware = serde_json::from_value(cfg["hardware"].clone()).map_err(|e| e.to_string())?;
    let context = args[2].parse::<usize>().map_err(|_| "context")?;
    let tile = args[3].parse::<usize>().map_err(|_| "tile")?;
    let count = args[4].parse::<usize>().map_err(|_| "groups")?;
    let asynchronous = args[5] == "1";
    if context == 0
        || context > 4096
        || tile == 0
        || tile > 64
        || count == 0
        || count > h.base.sms
        || tile * count > context
    {
        return Err("invalid probe size".into());
    }
    let hd = 64;
    let q = data(context * hd, 1);
    let k = data(context * hd, 2);
    let v = data(context * hd, 3);
    let mut m = Machine::new_concurrent(h.clone(), true, 1, false)?;
    m.alloc("q", q.clone())?;
    m.alloc("k", k.clone())?;
    m.alloc("v", v.clone())?;
    m.empty("out", context * hd)?;
    let full = cfg
        .get("probe_full")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let starts: Vec<_> = if full {
        (0..context).step_by(tile).collect()
    } else {
        ((context - count * tile)..context).step_by(tile).collect()
    };
    let mut waves = vec![];
    for chunk in starts.chunks(count) {
        waves.push(
            chunk
                .iter()
                .enumerate()
                .map(|(sm, start)| {
                    group(
                        *start,
                        tile.min(context - *start),
                        context,
                        hd,
                        tile,
                        true,
                        sm,
                        asynchronous,
                    )
                })
                .collect::<Vec<_>>(),
        );
    }
    let rf_kib = waves[0][0].rf_kib;
    let mut primitives = 0;
    let mut commands = 0;
    let mut bytes = 0;
    let mut total_json_bytes = 0;
    let mut max_wave_primitives = 0;
    let mut max_wave_commands = 0;
    for wave in &waves {
        let p: usize = wave
            .iter()
            .flat_map(|g| &g.commands)
            .filter(|c| !matches!(c, Command::Wait { .. }))
            .count();
        let c: usize = wave.iter().map(|g| g.commands.len()).sum();
        primitives += p;
        commands += c;
        max_wave_primitives = max_wave_primitives.max(p);
        max_wave_commands = max_wave_commands.max(c);
        struct Counter(usize);
        impl std::io::Write for Counter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0 += buf.len();
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut counter = Counter(0);
        serde_json::to_writer(
            &mut counter,
            &WaveRecord {
                record: "wave",
                groups: wave,
            },
        )
        .map_err(|e| e.to_string())?;
        bytes = bytes.max(counter.0 + 1);
        total_json_bytes += counter.0 + 1;
    }
    if let Some(path) = std::env::var_os("ATTENTION_EXPORT") {
        use std::io::Write;
        let file = fs::File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        let mut writer = std::io::BufWriter::new(file);
        let header = vnext_sim::v09_submission::Header {
            contract: vnext_sim::v09_submission::CONTRACT.into(),
            model: vnext_sim::MODEL.into(),
            hardware: h.clone(),
            shape: serde_json::from_value(cfg["shape"].clone()).map_err(|e| e.to_string())?,
            batch: 1,
            decode: false,
        };
        serde_json::to_writer(&mut writer, &header).map_err(|e| e.to_string())?;
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
        for wave in &waves {
            serde_json::to_writer(
                &mut writer,
                &WaveRecord {
                    record: "wave",
                    groups: wave,
                },
            )
            .map_err(|e| e.to_string())?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
    let start = Instant::now();
    let mut result = Ok(());
    let mut json_seconds = 0.;
    #[cfg(feature = "compact")]
    let imported = std::env::var_os("ATTENTION_PROGRAM");
    #[cfg(not(feature = "compact"))]
    let imported: Option<std::ffi::OsString> = None;
    if let Some(path) = imported {
        drop(waves);
        #[cfg(feature = "compact")]
        {
            let mut reader = vnext_sim::compact::Reader::open(Path::new(&path))?;
            if reader.header.contract != vnext_sim::v09_submission::CONTRACT
                || reader.header.model != vnext_sim::MODEL
                || serde_json::to_value(&reader.header.hardware).unwrap()
                    != serde_json::to_value(&h).unwrap()
            {
                return Err("probe program header mismatch".into());
            }
            loop {
                let json_start = Instant::now();
                let record = reader.next_record(false)?;
                json_seconds += json_start.elapsed().as_secs_f64();
                match record {
                    Some(vnext_sim::v09_submission::Record::Wave { groups }) => {
                        result = m.concurrent_wave(groups);
                        if result.is_err() {
                            break;
                        }
                    }
                    None => break,
                    _ => return Err("kernel probe accepts waves only".into()),
                }
            }
        }
        #[cfg(not(feature = "compact"))]
        let _ = path;
    } else {
        for wave in waves {
            let json_start = Instant::now();
            let encoded = serde_json::to_vec(&WaveRecord {
                record: "wave",
                groups: &wave,
            })
            .map_err(|e| e.to_string())?;
            drop(wave);
            if encoded.len() as u64 + 1 > vnext_sim::submission::MAX_LINE_BYTES {
                result = Err("wave exceeds JSONL line-byte admission limit".into());
                break;
            }
            let record: vnext_sim::v09_submission::Record =
                serde_json::from_slice(&encoded).map_err(|e| e.to_string())?;
            drop(encoded);
            let vnext_sim::v09_submission::Record::Wave { groups: wave } = record else {
                unreachable!()
            };
            json_seconds += json_start.elapsed().as_secs_f64();
            result = m.concurrent_wave(wave);
            if result.is_err() {
                break;
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let mut max_error = 0f64;
    let mut compared = 0;
    let mut correct = result.is_ok();
    let reference_start = Instant::now();
    if result.is_ok() {
        for row in (if full { 0 } else { context - count * tile })..context {
            let mut scores = vec![];
            for j in 0..=row {
                scores.push(
                    (0..hd)
                        .map(|d| q[row * hd + d] as f64 * k[j * hd + d] as f64)
                        .sum::<f64>()
                        / 8.,
                );
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let norm: f64 = weights.iter().sum();
            for d in 0..hd {
                let expected = (0..=row)
                    .map(|j| weights[j] * v[j * hd + d] as f64)
                    .sum::<f64>()
                    / norm;
                let actual = m.tensors[3].data[row * hd + d] as f64;
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                correct &= actual.is_finite() && error <= 1e-3 + 1e-3 * expected.abs();
                compared += 1;
            }
        }
    }
    let report = if result.is_ok() {
        Some(m.concurrent_report()?)
    } else {
        None
    };
    use sha2::{Digest, Sha256};
    let probe_hash = format!(
        "{:x}",
        Sha256::digest(include_str!("online_attention_probe.rs"))
    );
    let output = serde_json::json!({"model":vnext_sim::MODEL,"source_sha256":vnext_sim::source::hash(),"probe_sha256":probe_hash,"context":context,"tile":tile,"groups":count,"asynchronous":asynchronous,"primitives":primitives,"commands":commands,"wave_json_bytes":bytes,"total_wave_json_bytes":total_json_bytes,"rf_kib_per_group":rf_kib,"full_head":full,"max_wave_primitives":max_wave_primitives,"max_wave_commands":max_wave_commands,"host_seconds":seconds,"json_seconds":json_seconds,"reference_seconds":reference_start.elapsed().as_secs_f64(),"error":result.err(),"correct":correct,"compared":compared,"max_abs_error":max_error,"report":report});
    let path = Path::new(&args[6]);
    let file = fs::File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file, &output).map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string(&output).unwrap());
    Ok(())
}
