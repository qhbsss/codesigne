use vnext_sim::{
    baseline::{self, Fixture, Policy, Shape},
    hardware::Hardware,
    machine::Machine,
    reference,
};
#[test]
fn policies_preserve_complete_transformer_semantics() {
    let fixture = Fixture::new(Shape::smoke(), 37).unwrap();
    for decode in [false, true] {
        let expected = reference::run(&fixture, decode);
        for policy in [
            Policy {
                m: 3,
                n: 13,
                k: 17,
                vector_gemv: false,
                pack_transpose: false,
                split_k: 1,
                head_group: 1,
                persistent_keys: false,
                sh_rows: 0,
                sh_direct: false,
            },
            Policy {
                m: 64,
                n: 64,
                k: 64,
                vector_gemv: false,
                pack_transpose: true,
                split_k: 4,
                head_group: 1,
                persistent_keys: false,
                sh_rows: 0,
                sh_direct: false,
            },
            Policy {
                m: 32,
                n: 32,
                k: 64,
                vector_gemv: true,
                pack_transpose: false,
                split_k: 1,
                head_group: 1,
                persistent_keys: false,
                sh_rows: 0,
                sh_direct: false,
            },
            Policy {
                m: 32,
                n: 64,
                k: 64,
                vector_gemv: true,
                pack_transpose: true,
                split_k: 4,
                head_group: 1,
                persistent_keys: false,
                sh_rows: 0,
                sh_direct: false,
            },
        ] {
            let mut machine = Machine::new(Hardware::default(), true).unwrap();
            let actual = baseline::run_policy(&mut machine, &fixture, decode, policy).unwrap();
            reference::compare(&actual, &expected).unwrap();
            assert!(machine.report().power_pass);
        }
    }
}
#[test]
fn vector_fma_charges_accumulator_read_and_full_fma_energy() {
    use vnext_sim::isa::{Arg, Instr, Op};
    let mut machine = Machine::new(Hardware::default(), true).unwrap();
    let mut wave = vec![vec![]; 8];
    wave[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 32,
            value: 1.,
        },
        Instr::Vector {
            kind: Op::Fma,
            dst: 0,
            len: 32,
            a: Arg::Imm { value: 2. },
            b: Arg::Imm { value: 3. },
        },
    ];
    machine.wave(wave).unwrap();
    let r = machine.report();
    assert_eq!(r.stats.rf_bytes, 384); // fill write + accumulator read + FMA write
    assert!((machine.timing.power.dynamic_pj - (384. * 0.462 + 32. * 4. + 10. * 8.)).abs() < 1e-7);
}
#[test]
fn rectangular_capacity_and_invalid_policy_fail() {
    let fixture = Fixture::new(
        Shape {
            prefill: 64,
            d: 64,
            f: 256,
            heads: 4,
            ..Shape::smoke()
        },
        2,
    )
    .unwrap();
    let mut machine = Machine::new(
        Hardware {
            rf_kib: 32,
            ..Hardware::default()
        },
        false,
    )
    .unwrap();
    assert!(
        baseline::run_policy(
            &mut machine,
            &fixture,
            false,
            Policy {
                m: 64,
                n: 64,
                k: 256,
                vector_gemv: false,
                pack_transpose: false,
                split_k: 1,
                head_group: 1,
                persistent_keys: false,
                sh_rows: 0,
                sh_direct: false,
            }
        )
        .is_err()
    );
    assert!(Policy::square(0).validate().is_err());
    let legacy: Policy = serde_json::from_str(r#"{"m": 3, "n": 13, "k": 17}"#).unwrap();
    assert_eq!(legacy.head_group, 1);
    assert!(!legacy.persistent_keys);
    for head_group in [0, 33] {
        assert!(
            Policy {
                head_group,
                ..legacy
            }
            .validate()
            .is_err()
        );
    }
}

#[test]
fn scratch_lifetime_is_bounded_and_reuse_is_cold() {
    let mut machine = Machine::new(Hardware::default(), true).unwrap();
    let persistent = machine.alloc("persistent", vec![2.; 16]).unwrap();
    for _ in 0..1000 {
        let scratch = machine.empty("temporary", 16384).unwrap();
        assert!(machine.release_last(persistent).is_err());
        assert!(machine.tensors[scratch].data.iter().all(|v| v.is_nan()));
        machine.release_last(scratch).unwrap();
    }
    assert_eq!(machine.tensors.len(), 1);
    let report = machine.report();
    assert_eq!(report.hbm_allocated_bytes, 64);
    assert_eq!(report.hbm_peak_allocated_bytes, 256 + 65536);
    machine.reset();
    assert_eq!(machine.report().hbm_peak_allocated_bytes, 0);
}

#[test]
fn attention_stress_fixture_accepts_legal_split_and_packing() {
    use vnext_sim::baseline::FixtureProfile;
    let fixture =
        Fixture::new_profile(Shape::smoke(), 19, FixtureProfile::AttentionStress).unwrap();
    for decode in [false, true] {
        let mut machine = Machine::new(Hardware::default(), true).unwrap();
        let policy = Policy {
            m: 3,
            n: 13,
            k: 17,
            vector_gemv: true,
            pack_transpose: true,
            split_k: 8,
            head_group: 1,
            persistent_keys: false,
            sh_rows: 0,
            sh_direct: false,
        };
        let actual = baseline::run_policy(&mut machine, &fixture, decode, policy).unwrap();
        reference::compare(&actual, &reference::run(&fixture, decode)).unwrap();
    }
}

#[test]
fn grouped_heads_cover_tails_and_match_step_scheduler() {
    use vnext_sim::baseline::FixtureProfile;
    let fixture = Fixture::new_profile(
        Shape {
            layers: 2,
            d: 48,
            heads: 3,
            f: 97,
            prefill: 7,
            history: 11,
            steps: 2,
        },
        41,
        FixtureProfile::AttentionStress,
    )
    .unwrap();
    for decode in [false, true] {
        let expected = reference::run(&fixture, decode);
        for (group, split, vector, persistent) in [
            (2, 1, false, false),
            (3, 2, false, true),
            (4, 4, true, true),
            (1, 1, false, true),
        ] {
            let policy = Policy {
                m: 3,
                n: 13,
                k: 17,
                vector_gemv: vector,
                pack_transpose: true,
                split_k: split,
                head_group: group,
                persistent_keys: persistent,
                sh_rows: 0,
                sh_direct: false,
            };
            let mut reports = vec![];
            for cycle_reference in [false, true] {
                let mut machine = Machine::new(Hardware::default(), true).unwrap();
                machine.timing.cycle_reference = cycle_reference;
                let actual = baseline::run_policy(&mut machine, &fixture, decode, policy).unwrap();
                reference::compare(&actual, &expected).unwrap();
                reports.push(serde_json::to_value(machine.report()).unwrap());
            }
            assert_eq!(reports[0], reports[1]);
        }
    }
}
