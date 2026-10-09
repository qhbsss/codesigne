//! Intentionally incorrect attention negative control; never an eligible solution.
//! Standalone diagnostic copy of tensor math, independent of the simulator ISA.
use vnext_sim::{
    baseline::{Fixture, FixtureProfile, Outputs, Shape},
    reference,
};
fn matmul(x: &[f64], w: &[f32], bias: &[f32], rows: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.; rows * n];
    for i in 0..rows {
        for j in 0..n {
            out[i * n + j] = bias[j] as f64;
        }
        for kk in 0..k {
            let a = x[i * k + kk];
            for j in 0..n {
                out[i * n + j] += a * w[kk * n + j] as f64;
            }
        }
    }
    out
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
fn uniform_attention(f: &Fixture, decode: bool) -> Outputs {
    let s = f.shape;
    let d = s.d;
    let rows = if decode { 1 } else { s.prefill };
    let steps = if decode { s.steps } else { 1 };
    let mut keys: Vec<Vec<f64>> = f
        .layers
        .iter()
        .map(|l| {
            if decode {
                l.past_k.iter().map(|&v| v as f64).collect()
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
                l.past_v.iter().map(|&v| v as f64).collect()
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
            let qkv = matmul(&norm(&x, &l.g1, &l.b1, d), &l.qkv, &l.qb, rows, d, 3 * d);
            for i in 0..rows {
                keys[li].extend_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
                vals[li].extend_from_slice(&qkv[i * 3 * d + 2 * d..(i + 1) * 3 * d]);
            }
            result.keys[li].extend(keys[li][history * d..].iter().map(|&v| v as f32));
            result.values[li].extend(vals[li][history * d..].iter().map(|&v| v as f32));
            let mut att = vec![0.; rows * d];
            let hd = d / s.heads;
            for i in 0..rows {
                for h in 0..s.heads {
                    let visible = history + i + 1;
                    let mut score = vec![0.; visible];
                    // Forbidden shortcut: replace QK scores by zeros, while still
                    // returning full hidden and newly generated K/V tensors.
                    let max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for v in &mut score {
                        *v = (*v - max).exp();
                    }
                    let sum = score.iter().sum::<f64>();
                    for (j, v) in score.iter().enumerate() {
                        for k in 0..hd {
                            att[i * d + h * hd + k] += v / sum * vals[li][j * d + h * hd + k];
                        }
                    }
                }
            }
            let mut out = matmul(&att, &l.out, &l.ob, rows, d, d);
            for (y, xx) in out.iter_mut().zip(&x) {
                *y += xx;
            }
            let mut up = matmul(&norm(&out, &l.g2, &l.b2, d), &l.up, &l.ub, rows, d, s.f);
            for y in &mut up {
                *y = 0.5
                    * *y
                    * (1.
                        + (std::f64::consts::FRAC_2_PI.sqrt() * (*y + 0.044715 * *y * *y * *y))
                            .tanh());
            }
            x = matmul(&up, &l.down, &l.db, rows, s.f, d);
            for (y, z) in x.iter_mut().zip(&out) {
                *y += z;
            }
        }
        result.hidden.push(x.iter().map(|&v| v as f32).collect());
    }
    result
}

fn main() {
    let mut results = vec![];
    for seed in [7, 19, 37] {
        for strong in [false, true] {
            let fixture = Fixture::new_profile(
                Shape::main(),
                seed,
                if strong {
                    FixtureProfile::AttentionStress
                } else {
                    FixtureProfile::Legacy
                },
            )
            .unwrap();
            for decode in [false, true] {
                let expected = reference::run(&fixture, decode);
                let wrong = uniform_attention(&fixture, decode);
                let mut max_abs = 0f64;
                let mut max_scaled = 0f64;
                for (a, b) in wrong
                    .hidden
                    .iter()
                    .chain(&wrong.keys)
                    .chain(&wrong.values)
                    .zip(
                        expected
                            .hidden
                            .iter()
                            .chain(&expected.keys)
                            .chain(&expected.values),
                    )
                {
                    for (&x, &y) in a.iter().zip(b) {
                        let error = (x as f64 - y as f64).abs();
                        max_abs = max_abs.max(error);
                        max_scaled = max_scaled.max(error / (1e-3 + 1e-3 * (y as f64).abs()));
                    }
                }
                if strong {
                    assert!(
                        reference::compare(&wrong, &expected).is_err(),
                        "uniform attention bypassed attention_stress fixture"
                    );
                }
                results.push(serde_json::json!({"seed":seed,"strong_qk":strong,"decode":decode,
                    "incorrect_uniform_attention_passes":reference::compare(&wrong,&expected).is_ok(),
                    "max_abs":max_abs,"max_scaled":max_scaled}));
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

#[test]
fn stress_fixture_rejects_uniform_decode_attention() {
    for seed in [7, 19, 37] {
        let fixture = Fixture::new_profile(
            Shape {
                layers: 2,
                d: 64,
                heads: 4,
                f: 192,
                prefill: 7,
                history: 128,
                steps: 2,
            },
            seed,
            FixtureProfile::AttentionStress,
        )
        .unwrap();
        assert!(
            reference::compare(
                &uniform_attention(&fixture, true),
                &reference::run(&fixture, true)
            )
            .is_err()
        );
    }
}
