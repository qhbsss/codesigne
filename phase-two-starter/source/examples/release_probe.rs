//! Deliberately wrong algorithms for release acceptance negative controls.
//! This diagnostic never produces an eligible scored submission.
use vnext_sim::{
    baseline::{Fixture, FixtureProfile, Outputs, Shape},
    reference,
};

#[derive(Clone, Copy, Debug)]
enum Mutation {
    Uniform,
    Unmasked,
    NoScale,
    SkipFfn,
    SkipLastLayer,
    IgnorePast,
    NoBias,
    NoNorm,
    LinearGelu,
    GeluNoCubic,
    GeluRationalTanh,
    CorruptKv,
}
fn matmul(
    x: &[f64],
    w: &[f32],
    bias: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    mutation: Mutation,
) -> Vec<f64> {
    let mut out = vec![0.; rows * n];
    for i in 0..rows {
        for j in 0..n {
            out[i * n + j] = if matches!(mutation, Mutation::NoBias) {
                0.
            } else {
                bias[j] as f64
            };
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
fn norm(x: &[f64], g: &[f32], b: &[f32], d: usize, mutation: Mutation) -> Vec<f64> {
    if matches!(mutation, Mutation::NoNorm) {
        return x.to_vec();
    }
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
fn run(f: &Fixture, decode: bool, mutation: Mutation) -> Outputs {
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
            if matches!(mutation, Mutation::SkipLastLayer) && li + 1 == s.layers {
                continue;
            }
            let qkv = matmul(
                &norm(&x, &l.g1, &l.b1, d, mutation),
                &l.qkv,
                &l.qb,
                rows,
                d,
                3 * d,
                mutation,
            );
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
                    let visible = if matches!(mutation, Mutation::Unmasked) {
                        history + rows
                    } else {
                        history + i + 1
                    };
                    let mut score = vec![0.; visible];
                    for (j, v) in score.iter_mut().enumerate() {
                        for k in 0..hd {
                            *v += qkv[i * 3 * d + h * hd + k] * keys[li][j * d + h * hd + k];
                        }
                        if !matches!(mutation, Mutation::NoScale) {
                            *v /= (hd as f64).sqrt();
                        }
                        if matches!(mutation, Mutation::Uniform) {
                            *v = 0.;
                        }
                        if matches!(mutation, Mutation::IgnorePast) && j < history {
                            *v = f64::NEG_INFINITY;
                        }
                    }
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
            let mut out = matmul(&att, &l.out, &l.ob, rows, d, d, mutation);
            for (y, xx) in out.iter_mut().zip(&x) {
                *y += xx;
            }
            if matches!(mutation, Mutation::SkipFfn) {
                x = out;
                continue;
            }
            let mut up = matmul(
                &norm(&out, &l.g2, &l.b2, d, mutation),
                &l.up,
                &l.ub,
                rows,
                d,
                s.f,
                mutation,
            );
            for y in &mut up {
                if matches!(mutation, Mutation::LinearGelu) {
                    continue;
                }
                let cubic = if matches!(mutation, Mutation::GeluNoCubic) {
                    0.
                } else {
                    0.044715 * *y * *y * *y
                };
                let z = std::f64::consts::FRAC_2_PI.sqrt() * (*y + cubic);
                let t = if matches!(mutation, Mutation::GeluRationalTanh) {
                    let z = z.clamp(-3., 3.);
                    z * (27. + z * z) / (27. + 9. * z * z)
                } else {
                    z.tanh()
                };
                *y = 0.5 * *y * (1. + t);
            }
            x = matmul(&up, &l.down, &l.db, rows, s.f, d, mutation);
            for (y, z) in x.iter_mut().zip(&out) {
                *y += z;
            }
        }
        result.hidden.push(x.iter().map(|&v| v as f32).collect());
    }
    if matches!(mutation, Mutation::CorruptKv) {
        result.keys[0][0] += 0.1;
    }
    result
}

fn probe(shape: Shape, seeds: &[u64]) -> serde_json::Value {
    let mut results = vec![];
    for &seed in seeds {
        let fixture = Fixture::new_profile(shape, seed, FixtureProfile::AttentionStress).unwrap();
        for decode in [false, true] {
            let expected = reference::run(&fixture, decode);
            for mutation in [
                Mutation::Uniform,
                Mutation::Unmasked,
                Mutation::NoScale,
                Mutation::SkipFfn,
                Mutation::SkipLastLayer,
                Mutation::IgnorePast,
                Mutation::NoBias,
                Mutation::NoNorm,
                Mutation::LinearGelu,
                Mutation::GeluNoCubic,
                Mutation::GeluRationalTanh,
                Mutation::CorruptKv,
            ] {
                if (decode && matches!(mutation, Mutation::Unmasked))
                    || (!decode && matches!(mutation, Mutation::IgnorePast))
                {
                    continue;
                }
                let wrong = run(&fixture, decode, mutation);
                let comparison = reference::compare(&wrong, &expected);
                let approximation =
                    matches!(mutation, Mutation::GeluNoCubic | Mutation::GeluRationalTanh);
                if !approximation {
                    assert!(
                        comparison.is_err(),
                        "{mutation:?} seed {seed} decode {decode} accepted"
                    );
                }
                let max_scaled = wrong
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
                    .flat_map(|(a, b)| a.iter().zip(b))
                    .map(|(&a, &b)| (a as f64 - b as f64).abs() / (1e-3 + 1e-3 * (b as f64).abs()))
                    .fold(0., f64::max);
                results.push(serde_json::json!({"mutation":format!("{mutation:?}"),
                    "seed":seed,"decode":decode,"rejected":comparison.is_err(),
                    "approximation_probe":approximation,"max_scaled":max_scaled,
                    "reason":comparison.err()}));
            }
        }
    }
    serde_json::json!({"shape":shape,"results":results})
}
fn main() {
    let shape = if std::env::args().any(|a| a == "main") {
        Shape::main()
    } else {
        Shape {
            layers: 2,
            d: 96,
            heads: 4,
            f: 256,
            prefill: 17,
            history: 128,
            steps: 3,
        }
    };
    let seeds = if shape.d == 768 {
        &[7][..]
    } else {
        &[7, 19, 37][..]
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&probe(shape, seeds)).unwrap()
    );
}
#[test]
fn release_algorithm_negative_controls() {
    let result = probe(
        Shape {
            layers: 2,
            d: 96,
            heads: 4,
            f: 256,
            prefill: 17,
            history: 128,
            steps: 3,
        },
        &[7, 19, 37],
    );
    assert_eq!(result["results"].as_array().unwrap().len(), 66);
}
