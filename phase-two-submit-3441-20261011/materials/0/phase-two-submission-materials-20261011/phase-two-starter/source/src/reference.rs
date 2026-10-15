//! Independent f64 tensor math: no ISA, timing code, tile layout or reductions.
use crate::{
    Result,
    baseline::{Fixture, Outputs},
};
use serde::Serialize;
fn matmul(
    x: &[f64],
    w: &[f32],
    bias: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Vec<f64> {
    let mut out = vec![0.; rows * n];
    let chunk_rows = rows.div_ceil(threads.clamp(1, 8)).max(1);
    if threads == 1 || rows * k * n < 1_000_000 {
        matmul_rows(x, w, bias, k, n, &mut out);
    } else {
        std::thread::scope(|scope| {
            for (i, dst) in out.chunks_mut(chunk_rows * n).enumerate() {
                let src = &x[i * chunk_rows * k..i * chunk_rows * k + dst.len() / n * k];
                scope.spawn(move || matmul_rows(src, w, bias, k, n, dst));
            }
        });
    }
    out
}
fn matmul_rows(x: &[f64], w: &[f32], bias: &[f32], k: usize, n: usize, out: &mut [f64]) {
    for (dst, src) in out.chunks_exact_mut(n).zip(x.chunks_exact(k)) {
        for (v, &b) in dst.iter_mut().zip(bias) {
            *v = b as f64;
        }
        for (&a, wr) in src.iter().zip(w.chunks_exact(n)) {
            for (v, &b) in dst.iter_mut().zip(wr) {
                *v += a * b as f64;
            }
        }
    }
}
fn norm(x: &[f64], g: &[f32], b: &[f32], d: usize) -> Vec<f64> {
    let mut out = vec![0.; x.len()];
    for (src, dst) in x.chunks(d).zip(out.chunks_mut(d)) {
        let mean = src.iter().sum::<f64>() / d as f64;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        for j in 0..d {
            dst[j] = (src[j] - mean) / (var + 1e-5).sqrt() * g[j] as f64 + b[j] as f64;
        }
    }
    out
}
pub fn run(f: &Fixture, decode: bool) -> Outputs {
    run_batched(f, decode, 1, 1)
}
/// Batch is isolated in attention; only projections share weights across rows.
pub fn run_batched(f: &Fixture, decode: bool, batch: usize, threads: usize) -> Outputs {
    run_impl(f, decode, batch, threads, false)
}
/// Deliberately wrong algorithm, used only by the research negative-control probe.
#[cfg(feature = "explore")]
pub fn uniform_attention_control(
    f: &Fixture,
    decode: bool,
    batch: usize,
    threads: usize,
) -> Outputs {
    run_impl(f, decode, batch, threads, true)
}
fn run_impl(f: &Fixture, decode: bool, batch: usize, threads: usize, uniform: bool) -> Outputs {
    let s = f.shape;
    let d = s.d;
    let per_sample = if decode { 1 } else { s.prefill };
    let rows = batch * per_sample;
    let steps = if decode { s.steps } else { 1 };
    let mut keys: Vec<Vec<f64>> = f
        .layers
        .iter()
        .map(|l| {
            if decode {
                let mut data = Vec::with_capacity(l.past_k.len() + steps * batch * d);
                data.extend(l.past_k.iter().map(|&v| v as f64));
                data
            } else {
                vec![]
            }
        })
        .collect();
    let mut vals: Vec<Vec<f64>> = f
        .layers
        .iter()
        .map(|l| {
            if decode {
                let mut data = Vec::with_capacity(l.past_v.len() + steps * batch * d);
                data.extend(l.past_v.iter().map(|&v| v as f64));
                data
            } else {
                vec![]
            }
        })
        .collect();
    let mut result = Outputs {
        hidden: vec![],
        keys: vec![vec![]; s.layers],
        values: vec![vec![]; s.layers],
    };
    for step in 0..steps {
        let mut x: Vec<f64> = if decode { &f.inputs[step] } else { &f.prompt }
            .iter()
            .map(|&v| v as f64)
            .collect();
        let history = if decode { s.history + step } else { 0 };
        for (li, l) in f.layers.iter().enumerate() {
            let qkv = matmul(
                &norm(&x, &l.g1, &l.b1, d),
                &l.qkv,
                &l.qb,
                rows,
                d,
                3 * d,
                threads,
            );
            for token in 0..per_sample {
                for sample in 0..batch {
                    let i = sample * per_sample + token;
                    keys[li].extend_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
                    vals[li].extend_from_slice(&qkv[i * 3 * d + 2 * d..(i + 1) * 3 * d]);
                }
            }
            result.keys[li].extend(keys[li][history * batch * d..].iter().map(|&v| v as f32));
            result.values[li].extend(vals[li][history * batch * d..].iter().map(|&v| v as f32));
            let mut att = vec![0.; rows * d];
            let hd = d / s.heads;
            for i in 0..rows {
                for h in 0..s.heads {
                    let sample = i / per_sample;
                    let visible = history + i % per_sample + 1;
                    let mut score = vec![0.; visible];
                    for (j, v) in score.iter_mut().enumerate() {
                        for k in 0..hd {
                            *v += qkv[i * 3 * d + h * hd + k]
                                * keys[li][(j * batch + sample) * d + h * hd + k];
                        }
                        *v /= (hd as f64).sqrt();
                    }
                    let max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for v in &mut score {
                        *v = (*v - max).exp();
                    }
                    if uniform {
                        score.fill(1.);
                    }
                    let sum = score.iter().sum::<f64>();
                    for (j, v) in score.iter().enumerate() {
                        for k in 0..hd {
                            att[i * d + h * hd + k] +=
                                v / sum * vals[li][(j * batch + sample) * d + h * hd + k];
                        }
                    }
                }
            }
            let mut out = matmul(&att, &l.out, &l.ob, rows, d, d, threads);
            for (y, xx) in out.iter_mut().zip(&x) {
                *y += xx;
            }
            let mut up = matmul(
                &norm(&out, &l.g2, &l.b2, d),
                &l.up,
                &l.ub,
                rows,
                d,
                s.f,
                threads,
            );
            for y in &mut up {
                *y = 0.5
                    * *y
                    * (1.
                        + (std::f64::consts::FRAC_2_PI.sqrt() * (*y + 0.044715 * *y * *y * *y))
                            .tanh());
            }
            x = matmul(&up, &l.down, &l.db, rows, s.f, d, threads);
            for (y, z) in x.iter_mut().zip(&out) {
                *y += z;
            }
        }
        result.hidden.push(x.iter().map(|&v| v as f32).collect());
    }
    result
}
#[derive(Default, Serialize, Debug)]
pub struct Comparison {
    pub elements: usize,
    pub max_abs: f64,
    pub max_scaled: f64,
}
pub fn compare(actual: &Outputs, reference: &Outputs) -> Result<Comparison> {
    let mut stats = Comparison::default();
    for (name, aa, bb) in [
        ("hidden", &actual.hidden, &reference.hidden),
        ("K", &actual.keys, &reference.keys),
        ("V", &actual.values, &reference.values),
    ] {
        if aa.len() != bb.len() {
            return Err(format!("{name} output count mismatch"));
        }
        for (a, b) in aa.iter().zip(bb) {
            if a.len() != b.len() {
                return Err(format!("{name} output size mismatch"));
            }
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                let error = (x as f64 - y as f64).abs();
                let scaled = error / (1e-3 + 1e-3 * (y as f64).abs());
                if !x.is_finite() || !y.is_finite() || scaled > 1. {
                    return Err(format!("{name}[{i}] mismatch: {x} vs {y}"));
                }
                stats.elements += 1;
                stats.max_abs = stats.max_abs.max(error);
                stats.max_scaled = stats.max_scaled.max(scaled);
            }
        }
    }
    Ok(stats)
}
