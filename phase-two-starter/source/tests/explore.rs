#![cfg(feature = "explore")]
use vnext_sim::{
    baseline::{Fixture, FixtureProfile, Policy, Shape},
    explore::{Config, run_batched},
    hardware::Hardware,
    isa::{Arg, Instr, Op},
    machine::Machine,
    reference,
};
fn config(decode: bool) -> Config {
    Config {
        shape: Shape::smoke(),
        batch: 3,
        decode,
        hardware: Hardware {
            sms: 4,
            rf_kib: 64,
            sh_kib: 64,
            sh_banks: 8,
            sh_tc_bw: 64,
            ..Hardware::default()
        },
        policy: Policy {
            head_group: 2,
            persistent_keys: true,
            pack_transpose: true,
            ..Policy::square(8)
        },
        fixture_profile: FixtureProfile::AttentionStress,
        reference_threads: 1,
    }
}
#[test]
fn batch_isolation_matches_separately_executed_samples() {
    for decode in [false, true] {
        for persistent in [false, true] {
            let mut c = config(decode);
            c.policy.persistent_keys = persistent;
            let fixture = c.fixture(77).unwrap();
            let expected = reference::run_batched(&fixture, decode, c.batch, 1);
            let mut machine = Machine::new(c.hardware.clone(), true).unwrap();
            let actual = run_batched(&mut machine, &fixture, decode, c.policy, c.batch).unwrap();
            reference::compare(&actual, &expected).unwrap();
            let d = c.shape.d;
            let rows = if decode { 1 } else { c.shape.prefill };
            for sample in 0..c.batch {
                let mut single = Fixture::new(c.shape, 0).unwrap();
                single.layers = fixture.layers.clone();
                for l in &mut single.layers {
                    l.past_k = l
                        .past_k
                        .chunks(c.batch * d)
                        .flat_map(|v| v[sample * d..(sample + 1) * d].iter().copied())
                        .collect();
                    l.past_v = l
                        .past_v
                        .chunks(c.batch * d)
                        .flat_map(|v| v[sample * d..(sample + 1) * d].iter().copied())
                        .collect();
                }
                if decode {
                    single.inputs = fixture
                        .inputs
                        .iter()
                        .map(|v| v[sample * d..(sample + 1) * d].to_vec())
                        .collect();
                } else {
                    single.prompt =
                        fixture.prompt[sample * rows * d..(sample + 1) * rows * d].to_vec();
                }
                let oracle = reference::run(&single, decode);
                for (a, b) in expected.hidden.iter().zip(&oracle.hidden) {
                    assert_eq!(&a[sample * rows * d..(sample + 1) * rows * d], b);
                }
                for (a, b) in expected
                    .keys
                    .iter()
                    .chain(&expected.values)
                    .zip(oracle.keys.iter().chain(&oracle.values))
                {
                    let extracted: Vec<_> = a
                        .chunks(c.batch * d)
                        .flat_map(|v| v[sample * d..(sample + 1) * d].iter().copied())
                        .collect();
                    assert_eq!(&extracted, b);
                }
            }
        }
    }
}
#[test]
fn parallel_reference_is_bit_identical() {
    let mut c = config(false);
    c.shape.d = 128;
    c.shape.f = 512;
    c.shape.prefill = 16;
    let f = c.fixture(9).unwrap();
    let one = reference::run_batched(&f, false, 3, 1);
    let four = reference::run_batched(&f, false, 3, 4);
    assert_eq!(one.hidden, four.hidden);
    assert_eq!(one.keys, four.keys);
    assert_eq!(one.values, four.values);
}
fn hw(p: usize, q: usize) -> Hardware {
    Hardware {
        sms: 2,
        p,
        q,
        vector: 8,
        sfu: 1,
        rf_kib: 64,
        hbm_channels: 1,
        hbm_queue: 32,
        sm_noc: 16,
        global_noc: 64,
        ..Hardware::default()
    }
}
#[test]
fn all_array_shapes_execute_and_disabled_tc_rejects_mma() {
    for p in [1, 2, 4, 8, 16, 32, 64] {
        for q in [1, 2, 4, 8, 16, 32, 64] {
            if !(16..=512).contains(&(p * q)) {
                continue;
            }
            let mut m = Machine::new(hw(p, q), true).unwrap();
            let k = 8;
            let a = 0;
            let b = p * k;
            let c = b + k * q;
            m.wave(vec![
                vec![
                    Instr::Fill {
                        dst: a,
                        len: p * k,
                        value: 1.,
                    },
                    Instr::Fill {
                        dst: b,
                        len: k * q,
                        value: 1.,
                    },
                    Instr::Fill {
                        dst: c,
                        len: p * q,
                        value: 0.,
                    },
                    Instr::Mma {
                        a,
                        b,
                        c,
                        m: p,
                        n: q,
                        k,
                    },
                ],
                vec![],
            ])
            .unwrap();
            assert_eq!(m.report().stats.tc_physical_fmas, (p * q * k) as u64);
        }
    }
    let mut m = Machine::new(hw(0, 0), false).unwrap();
    assert!(
        m.wave(vec![
            vec![Instr::Mma {
                a: 0,
                b: 64,
                c: 128,
                m: 1,
                n: 1,
                k: 1
            }],
            vec![]
        ])
        .is_err()
    );
    assert!(hw(0, 8).validate().is_err());
}
#[test]
fn skinny_array_has_paid_input_bandwidth() {
    for tier in 1..=3 {
        let mut h = hw(1, 64);
        h.rf_kib = 32;
        h.rf_tier = tier;
        let mut m = Machine::new(h.clone(), false).unwrap();
        m.wave(vec![
            vec![Instr::Mma {
                a: 0,
                b: 16,
                c: 1040,
                m: 1,
                n: 64,
                k: 16,
            }],
            vec![],
        ])
        .unwrap();
        let step = 260usize.div_ceil(h.read_bw()) as u64;
        let expected = 1
            + 256usize.div_ceil(h.read_bw()) as u64
            + 2
            + 16 * step
            + 2
            + 1
            + 63
            + 256usize.div_ceil(h.write_bw()) as u64
            + 2
            + 1;
        assert_eq!(m.report().stats.cycles, expected);
    }
}
#[test]
fn longest_sfu_fits_bounded_power_horizon() {
    let mut h = hw(0, 0);
    h.rf_kib = 256;
    let mut m = Machine::new(h, true).unwrap();
    m.wave(vec![
        vec![
            Instr::Fill {
                dst: 0,
                len: 8192,
                value: 0.,
            },
            Instr::Vector {
                kind: Op::Exp,
                dst: 8192,
                len: 8192,
                a: Arg::reg(0),
                b: Arg::Imm { value: 0. },
            },
        ],
        vec![],
    ])
    .unwrap();
    let r = m.report();
    assert!(r.stats.cycles > 150_000);
    assert!(r.short_power_w.is_finite());
}
#[test]
fn configurations_reject_before_large_allocation() {
    let mut c = config(true);
    c.batch = usize::MAX;
    assert!(c.fixture(1).is_err());
    c.batch = 32;
    c.shape = Shape {
        layers: 8,
        d: 1536,
        heads: 12,
        f: 6144,
        prefill: 2048,
        history: 4096,
        steps: 32,
    };
    assert!(c.fixture(1).is_err());
}

#[test]
fn model_weights_do_not_depend_on_batch_or_context() {
    let a = config(false);
    let mut b = a.clone();
    b.batch = 7;
    b.shape.prefill = 17;
    b.shape.history = 63;
    b.shape.steps = 1;
    let x = a.fixture(99).unwrap();
    let y = b.fixture(99).unwrap();
    for (u, v) in x.layers.iter().zip(&y.layers) {
        assert_eq!(u.qkv, v.qkv);
        assert_eq!(u.out, v.out);
        assert_eq!(u.up, v.up);
        assert_eq!(u.down, v.down);
    }
}
#[test]
fn wide_shared_array_pays_bank_service_even_with_wide_interface() {
    use vnext_sim::isa::View;
    for banks in [8, 16, 32, 64] {
        let mut h = hw(1, 64);
        h.sh_kib = 64;
        h.sh_banks = banks;
        h.sh_tc_bw = 256;
        let mut m = Machine::new(h, false).unwrap();
        m.wave(vec![
            vec![Instr::MmaShared {
                a: 0,
                b: View::contiguous(0, 16, 64),
                c: 16,
                m: 1,
                n: 64,
                k: 16,
            }],
            vec![],
        ])
        .unwrap();
        let r = m.report();
        assert_eq!(r.stats.sh_bank_stall_cycles, 16 * (64 / banks as u64 - 1));
        assert_eq!(r.stats.sh_tc_bytes, 4096);
    }
}

#[test]
fn research_model_cannot_use_frozen_score_contract() {
    use vnext_sim::submission::{CONTRACT, Header};
    let h = Header {
        contract: CONTRACT.into(),
        model: vnext_sim::MODEL.into(),
        hardware: Hardware::default(),
        shape: Shape::main(),
        decode: false,
    };
    assert!(h.validate().unwrap_err().contains("no frozen static"));
}

#[test]
fn inactive_scenario_tensors_do_not_allocate_or_change_active_random_stream() {
    let mut c = config(false);
    c.shape.history = 4096;
    c.shape.steps = 32;
    let f = c.fixture(19).unwrap();
    assert!(
        f.layers
            .iter()
            .all(|l| l.past_k.is_empty() && l.past_v.is_empty())
    );
    assert!(f.inputs.iter().all(Vec::is_empty));
    assert_eq!(f.prompt.len(), c.batch * c.shape.prefill * c.shape.d);
    // Independent indexed form of the previous sequential generator.
    let value_at = |index: usize| {
        let mut z = (19u64 ^ 0xd1b54a32d192ed03)
            .wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(index as u64 + 1));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8388608. - 1.) * 0.5
    };
    let skipped = 2 * c.shape.layers * c.batch * c.shape.history * c.shape.d;
    for (i, value) in f.prompt.iter().enumerate() {
        assert_eq!(value.to_bits(), value_at(skipped + i).to_bits());
    }
    c.decode = true;
    let f = c.fixture(19).unwrap();
    assert!(f.prompt.is_empty());
    assert_eq!(f.inputs[0].len(), c.batch * c.shape.d);
    let skipped = skipped + c.batch * c.shape.prefill * c.shape.d;
    for (i, value) in f.inputs.iter().flatten().enumerate() {
        assert_eq!(value.to_bits(), value_at(skipped + i).to_bits());
    }
}
