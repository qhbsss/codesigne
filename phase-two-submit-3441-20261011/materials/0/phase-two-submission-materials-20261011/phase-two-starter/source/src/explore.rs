//! Versioned research harness; not a student grading or official-score entry point.
use crate::{
    Result,
    baseline::{Builder, Fixture, FixtureProfile, Matrix, Outputs, Policy, Shape},
    hardware::Hardware,
    machine::Machine,
};
use serde::{Deserialize, Serialize};
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
}
fn one() -> usize {
    1
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        self.shape.validate()?;
        self.hardware.validate()?;
        self.policy.validate()?;
        if !(1..=32).contains(&self.batch) || !(1..=8).contains(&self.reference_threads) {
            return Err("batch 1..32; reference_threads 1..8".into());
        }
        let s = self.shape;
        let weights = 4 * s.layers * (4 * s.d * s.d + 2 * s.d * s.f + 9 * s.d + s.f);
        let kv = 8
            * s.layers
            * self.batch
            * (if self.decode {
                s.history + s.steps
            } else {
                s.prefill
            })
            * s.d;
        if weights + kv > 1536 * 1024 * 1024 {
            return Err("essential tensors exceed research 1.5 GiB input budget".into());
        }
        let tokens = self.batch * if self.decode { s.steps } else { s.prefill };
        let work = s.layers * (4 * s.d * s.d + 2 * s.d * s.f) * tokens;
        if work > 40_000_000_000 {
            return Err("projection work exceeds research budget".into());
        }
        Ok(())
    }
    pub fn fixture(&self, seed: u64) -> Result<Fixture> {
        self.validate()?;
        // The parameter RNG must not depend on batch, prompt or history length.
        // Otherwise later layers silently use different weights in each workload.
        let weight_shape = Shape {
            prefill: 1,
            history: 1,
            steps: 1,
            ..self.shape
        };
        let mut f = Fixture::new_profile(weight_shape, seed, self.fixture_profile)?;
        f.shape = self.shape;
        // Independent sample data; weights remain shared and are allocated once.
        let mut state = seed ^ 0xd1b54a32d192ed03;
        let mut values = |n: usize, scale: f32, needed: bool| -> Vec<f32> {
            if !needed {
                // SplitMix is counter based: preserve the exact random stream
                // without allocating scenario-inaccessible tensors.
                state = state.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(n as u64));
                return vec![];
            }
            (0..n)
                .map(|_| {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    z ^= z >> 31;
                    ((z >> 40) as f32 / 8388608. - 1.) * scale
                })
                .collect()
        };
        for layer in &mut f.layers {
            layer.past_k = values(
                self.batch * self.shape.history * self.shape.d,
                if matches!(self.fixture_profile, FixtureProfile::AttentionStress) {
                    1.
                } else {
                    0.25
                },
                self.decode,
            );
            layer.past_v = values(
                self.batch * self.shape.history * self.shape.d,
                0.25,
                self.decode,
            );
        }
        f.prompt = values(
            self.batch * self.shape.prefill * self.shape.d,
            0.5,
            !self.decode,
        );
        f.inputs = (0..self.shape.steps)
            .map(|_| values(self.batch * self.shape.d, 0.5, self.decode))
            .collect();
        Ok(f)
    }
}
pub fn run_batched(
    machine: &mut Machine,
    fixture: &Fixture,
    decode: bool,
    policy: Policy,
    batch: usize,
) -> Result<Outputs> {
    policy.validate()?;
    let shape = fixture.shape;
    let d = shape.d;
    let f = shape.f;
    let per_sample = if decode { 1 } else { shape.prefill };
    let rows = batch * per_sample;
    let steps = if decode { shape.steps } else { 1 };
    let mut weights = vec![];
    let mut caches = vec![];
    for (i, l) in fixture.layers.iter().enumerate() {
        let mut ids = vec![];
        for (name, data) in [
            ("qkv", &l.qkv),
            ("qb", &l.qb),
            ("out", &l.out),
            ("ob", &l.ob),
            ("up", &l.up),
            ("ub", &l.ub),
            ("down", &l.down),
            ("db", &l.db),
            ("g1", &l.g1),
            ("b1", &l.b1),
            ("g2", &l.g2),
            ("b2", &l.b2),
        ] {
            ids.push(machine.bind_input(&format!("layer{i}.{name}"), data, data.len())?);
        }
        weights.push(ids);
        let capacity = if decode {
            shape.history + steps
        } else {
            per_sample
        };
        let (k, v) = if decode {
            (
                machine.bind_input(&format!("layer{i}.past_k"), &l.past_k, capacity * batch * d)?,
                machine.bind_input(&format!("layer{i}.past_v"), &l.past_v, capacity * batch * d)?,
            )
        } else {
            (
                machine.empty("K", capacity * batch * d)?,
                machine.empty("V", capacity * batch * d)?,
            )
        };
        let kt = if policy.persistent_keys {
            let kt = machine.empty("persistent_KT", capacity * batch * d)?;
            if decode {
                // Initial cache format conversion is part of measured execution.
                Builder { machine, policy }.pack_into(
                    Matrix {
                        tensor: k,
                        rows: batch * d,
                        cols: shape.history,
                        base: 0,
                        rs: 1,
                        cs: batch * d,
                    },
                    Matrix {
                        tensor: kt,
                        rows: batch * d,
                        cols: shape.history,
                        base: 0,
                        rs: capacity,
                        cs: 1,
                    },
                )?;
            }
            Some(kt)
        } else {
            None
        };
        caches.push((k, v, kt, capacity));
    }
    let mut outputs = Outputs {
        hidden: vec![],
        keys: vec![vec![]; shape.layers],
        values: vec![vec![]; shape.layers],
    };
    for step in 0..steps {
        // Future teacher-forced inputs do not exist in machine memory until
        // all preceding outputs and KV stores have completed at wave fences.
        let input = if decode {
            fixture.inputs[step].clone()
        } else {
            fixture.prompt.clone()
        };
        let x = machine.bind_input(&format!("input{step}"), &input, input.len())?;
        let history = if decode { shape.history + step } else { 0 };
        let context = history + per_sample;
        for layer in 0..shape.layers {
            let w = &weights[layer];
            let (kcache, vcache, kt, capacity) = caches[layer];
            let ln = machine.empty("ln", rows * d)?;
            let qkv = machine.empty("qkv", rows * 3 * d)?;
            let att = machine.empty("attention", rows * d)?;
            let out = machine.empty("projected", rows * d)?;
            let ln2 = machine.empty("ln2", rows * d)?;
            let up = machine.empty("up", rows * f)?;
            let down = machine.empty("down", rows * d)?;
            let mut b = Builder { machine, policy };
            b.norm(x, ln, w[8], w[9], rows, d)?;
            b.gemm(
                Matrix::plain(ln, rows, d),
                Matrix::plain(w[0], d, 3 * d),
                Matrix::plain(qkv, rows, 3 * d),
            )?;
            b.bias(qkv, w[1], rows, 3 * d)?;
            for (offset, cache) in [(d, kcache), (2 * d, vcache)] {
                for sample in 0..batch {
                    b.copy(
                        Matrix {
                            tensor: qkv,
                            rows: per_sample,
                            cols: d,
                            base: sample * per_sample * 3 * d + offset,
                            rs: 3 * d,
                            cs: 1,
                        },
                        Matrix {
                            tensor: cache,
                            rows: per_sample,
                            cols: d,
                            base: (history * batch + sample) * d,
                            rs: batch * d,
                            cs: 1,
                        },
                    )?;
                }
            }
            if let Some(kt) = kt {
                b.pack_into(
                    Matrix {
                        tensor: kcache,
                        rows: batch * d,
                        cols: per_sample,
                        base: history * batch * d,
                        rs: 1,
                        cs: batch * d,
                    },
                    Matrix {
                        tensor: kt,
                        rows: batch * d,
                        cols: per_sample,
                        base: history,
                        rs: capacity,
                        cs: 1,
                    },
                )?;
            }
            let hd = d / shape.heads;
            for sample in 0..batch {
                for first in (0..shape.heads).step_by(policy.head_group) {
                    let mut scores = vec![];
                    let mut qk = vec![];
                    let mut av = vec![];
                    for h in first..(first + policy.head_group).min(shape.heads) {
                        let score = b.machine.empty("score", per_sample * context)?;
                        scores.push(score);
                        let key = if let Some(kt) = kt {
                            Matrix {
                                tensor: kt,
                                rows: hd,
                                cols: context,
                                base: (sample * d + h * hd) * capacity,
                                rs: capacity,
                                cs: 1,
                            }
                        } else {
                            Matrix {
                                tensor: kcache,
                                rows: hd,
                                cols: context,
                                base: sample * d + h * hd,
                                rs: 1,
                                cs: batch * d,
                            }
                        };
                        qk.push((
                            Matrix {
                                tensor: qkv,
                                rows: per_sample,
                                cols: hd,
                                base: sample * per_sample * 3 * d + h * hd,
                                rs: 3 * d,
                                cs: 1,
                            },
                            key,
                            Matrix::plain(score, per_sample, context),
                        ));
                        av.push((
                            Matrix::plain(score, per_sample, context),
                            Matrix {
                                tensor: vcache,
                                rows: context,
                                cols: hd,
                                base: sample * d + h * hd,
                                rs: batch * d,
                                cs: 1,
                            },
                            Matrix {
                                tensor: att,
                                rows: per_sample,
                                cols: hd,
                                base: sample * per_sample * d + h * hd,
                                rs: d,
                                cs: 1,
                            },
                        ));
                    }
                    b.gemm_many(&qk)?;
                    b.softmax_many(&scores, per_sample, context, history, hd)?;
                    b.gemm_many(&av)?;
                    for score in scores.into_iter().rev() {
                        b.machine.release_last(score)?;
                    }
                }
            }
            b.gemm(
                Matrix::plain(att, rows, d),
                Matrix::plain(w[2], d, d),
                Matrix::plain(out, rows, d),
            )?;
            b.bias(out, w[3], rows, d)?;
            b.residual(x, out, rows, d)?;
            b.norm(out, ln2, w[10], w[11], rows, d)?;
            b.gemm(
                Matrix::plain(ln2, rows, d),
                Matrix::plain(w[4], d, f),
                Matrix::plain(up, rows, f),
            )?;
            b.bias(up, w[5], rows, f)?;
            b.gelu(up, rows, f)?;
            b.gemm(
                Matrix::plain(up, rows, f),
                Matrix::plain(w[6], f, d),
                Matrix::plain(down, rows, d),
            )?;
            b.bias(down, w[7], rows, d)?;
            b.residual(out, down, rows, d)?;
            // Reclaim whole layer scratch, using a paid copy to the live input/output.
            b.copy(Matrix::plain(down, rows, d), Matrix::plain(x, rows, d))?;
            while b.machine.tensors.len() > ln {
                b.machine.release_last(b.machine.tensors.len() - 1)?;
            }
            if b.machine.functional {
                outputs.keys[layer].extend_from_slice(
                    &b.machine.tensors[kcache].data[history * batch * d..context * batch * d],
                );
                outputs.values[layer].extend_from_slice(
                    &b.machine.tensors[vcache].data[history * batch * d..context * batch * d],
                );
            }
        }
        if machine.functional {
            outputs.hidden.push(machine.tensors[x].data.clone());
        }
        // Explicit STEP.COMMIT control issue, with no outstanding operations.
        use crate::submission::{Checkpoint, OutputView};
        machine.checkpoint(Checkpoint {
            hidden: OutputView {
                tensor: x,
                base: 0,
                len: rows * d,
            },
            keys: caches
                .iter()
                .map(|&(k, _, _, _)| OutputView {
                    tensor: k,
                    base: history * batch * d,
                    len: rows * d,
                })
                .collect(),
            values: caches
                .iter()
                .map(|&(_, v, _, _)| OutputView {
                    tensor: v,
                    base: history * batch * d,
                    len: rows * d,
                })
                .collect(),
        })?;
    }
    Ok(outputs)
}
