use vnext_sim::{
    baseline::{self, Fixture, Shape},
    hardware::Hardware,
    isa::{Arg, Instr, Op, View},
    machine::Machine,
    reference,
};
fn load(tensor: usize, base: usize, len: usize) -> Instr {
    Instr::Load {
        tensor,
        global: View::contiguous(base, 1, len),
        local: View::contiguous(0, 1, len),
    }
}
fn machine() -> Machine {
    Machine::new(Hardware::default(), true).unwrap()
}
// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn hand_calculated_single_line() {
    let mut m = machine();
    let t = m.alloc("x", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0].push(load(t, 0, 16));
    m.wave(w).unwrap();
    let r = m.report();
    // 1 issue + 6 startup; NoC 1+6; HBM 250+2; NoC 1+6;
    // RF write 1+2; one global wave fence = 277.
    assert_eq!(r.stats.cycles, 277);
    assert_eq!(r.stats.hbm_reads, 1);
    assert_eq!(r.stats.noc_bytes, 72);
    let expected = 8. + 20. + 8. + 2. + 8. + 64. * 150. + 72. * 2. + 64. * 0.462 + 8. * 8.;
    assert!((m.timing.power.dynamic_pj - expected).abs() < 1e-7);
}
// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn channel_contention_is_real() {
    let mut times = vec![];
    for distinct in [false, true] {
        let mut m = machine();
        let a = m.alloc("a", vec![1.; 16]).unwrap();
        let b = m.alloc("b", vec![2.; 16]).unwrap();
        let mut w = vec![vec![]; 8];
        w[0].push(load(a, 0, 16));
        w[1].push(load(if distinct { b } else { a }, 0, 16));
        m.wave(w).unwrap();
        times.push(m.timing.stats.cycles);
    }
    assert_eq!(times, vec![279, 277]);
}
#[test]
fn sixteen_element_coalescing_and_misalignment() {
    for (base, len, expected) in [(0, 16, 1), (1, 16, 2), (0, 17, 2)] {
        let mut m = machine();
        let a = m.alloc("a", vec![1.; 64]).unwrap();
        let mut w = vec![vec![]; 8];
        w[0].push(load(a, base, len));
        m.wave(w).unwrap();
        assert_eq!(m.timing.stats.hbm_reads, expected);
    }
}
#[test]
fn no_cross_batch_coalescing() {
    let mut m = machine();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0].push(Instr::Load {
        tensor: a,
        global: View {
            base: 0,
            rows: 2,
            cols: 16,
            row_stride: 0,
            col_stride: 1,
        },
        local: View::contiguous(0, 2, 16),
    });
    m.wave(w).unwrap();
    assert_eq!(m.timing.stats.hbm_reads, 2);
}
#[test]
fn cross_sm_raw_war_and_waw_rejected() {
    for other_write in [false, true] {
        let mut m = machine();
        let a = m.alloc("a", vec![1.; 16]).unwrap();
        let mut w = vec![vec![]; 8];
        w[0] = vec![
            Instr::Fill {
                dst: 0,
                len: 16,
                value: 2.,
            },
            Instr::Store {
                tensor: a,
                global: View::contiguous(0, 1, 16),
                local: View::contiguous(0, 1, 16),
            },
        ];
        w[1] = if other_write {
            w[0].clone()
        } else {
            vec![load(a, 0, 16)]
        };
        assert!(m.wave(w).unwrap_err().contains("race"));
    }
}
#[test]
fn barrier_enables_dependency_and_rf_does_not_survive() {
    let mut m = machine();
    let a = m.empty("a", 16).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 16,
            value: 2.,
        },
        Instr::Store {
            tensor: a,
            global: View::contiguous(0, 1, 16),
            local: View::contiguous(0, 1, 16),
        },
    ];
    m.wave(w).unwrap();
    let mut w = vec![vec![]; 8];
    w[1].push(load(a, 0, 16));
    m.wave(w).unwrap();
    let mut w = vec![vec![]; 8];
    w[1].push(Instr::Vector {
        kind: Op::Square,
        dst: 0,
        len: 1,
        a: Arg::reg(0),
        b: Arg::Imm { value: 0. },
    });
    assert!(m.wave(w).unwrap_err().contains("uninitialized"));
}
#[test]
fn uninitialized_hbm_and_bounds_fail() {
    let mut m = machine();
    let a = m.empty("a", 16).unwrap();
    let mut w = vec![vec![]; 8];
    w[0].push(load(a, 0, 16));
    assert!(m.wave(w).is_err());
    let mut w = vec![vec![]; 8];
    w[0].push(load(a, usize::MAX, 16));
    assert!(m.wave(w).is_err());
}
#[test]
fn bad_hardware_and_unknown_fields_fail() {
    let mut h = Hardware {
        rf_tier: 0,
        ..Hardware::default()
    };
    assert!(h.validate().is_err());
    h = Hardware::default();
    h.sms = 1024;
    assert!(h.validate().is_err());
    let mut json = serde_json::to_value(Hardware::default()).unwrap();
    json["cache_mib"] = 1.into();
    assert!(serde_json::from_value::<Hardware>(json).is_err());
}
#[test]
fn queue_and_frontend_are_bounded_under_fragmentation() {
    let h = Hardware {
        hbm_channels: 1,
        hbm_queue: 64,
        ..Hardware::default()
    };
    let mut m = Machine::new(h, true).unwrap();
    let a = m.alloc("a", vec![1.; 64 * 4096]).unwrap();
    let mut w = vec![vec![]; 8];
    for p in &mut w {
        p.push(Instr::Load {
            tensor: a,
            global: View {
                base: 0,
                rows: 1,
                cols: 4096,
                row_stride: 0,
                col_stride: 64,
            },
            local: View::contiguous(0, 1, 4096),
        });
    }
    m.wave(w).unwrap();
    assert_eq!(m.timing.stats.hbm_reads, 8 * 4096);
    assert!(m.timing.stats.max_hbm_inflight <= 64);
    assert!(m.timing.stats.max_frontend <= 32);
    assert!(m.timing.stats.max_live_events <= 256);
    assert!(m.timing.stats.hbm_queue_full_sm_cycles > 0);
}
#[test]
fn complete_transformer_hidden_and_each_layer_kv() {
    for seed in [0, 7, 91] {
        let f = Fixture::new(Shape::smoke(), seed).unwrap();
        for decode in [false, true] {
            let mut m = machine();
            let actual = baseline::run(&mut m, &f, decode, 16).unwrap();
            let cmp = reference::compare(&actual, &reference::run(&f, decode)).unwrap();
            assert!(cmp.elements > 0);
            assert!(cmp.max_abs < 1e-5);
            assert!(m.report().power_pass);
        }
    }
}
#[test]
fn tile_tail_and_estimate_have_identical_timing() {
    let f = Fixture::new(
        Shape {
            d: 24,
            heads: 3,
            f: 37,
            ..Shape::smoke()
        },
        4,
    )
    .unwrap();
    let mut counts = vec![];
    for functional in [true, false] {
        let mut m = Machine::new(Hardware::default(), functional).unwrap();
        let actual = baseline::run(&mut m, &f, false, 32).unwrap();
        if functional {
            reference::compare(&actual, &reference::run(&f, false)).unwrap();
        }
        counts.push(m.timing.stats.cycles);
    }
    assert_eq!(counts[0], counts[1]);
}
#[test]
fn repeated_simulations_are_deterministic() {
    let f = Fixture::new(Shape::smoke(), 7).unwrap();
    let mut reports = vec![];
    for _ in 0..2 {
        let mut m = machine();
        baseline::run(&mut m, &f, true, 32).unwrap();
        reports.push(serde_json::to_string(&m.report()).unwrap());
    }
    assert_eq!(reports[0], reports[1]);
}

#[test]
fn event_jumps_match_cycle_step_scheduler() {
    // Differential check of temporal batching. The independent checks of the
    // resource model remain the hand calculations and numerical oracle above.
    for seed in 0..32 {
        let h = Hardware {
            sms: 4,
            hbm_channels: if seed % 2 == 0 { 1 } else { 4 },
            hbm_queue: 64,
            sm_noc: 32,
            global_noc: 128,
            ..Hardware::default()
        };
        let mut reports = vec![];
        for cycle_reference in [false, true] {
            let mut m = Machine::new(h.clone(), true).unwrap();
            m.timing.cycle_reference = cycle_reference;
            let a = m.alloc("input", vec![1.; 65536]).unwrap();
            let b = m.empty("output", 4 * 4096).unwrap();
            let mut w = vec![vec![]; 4];
            for (sm, p) in w.iter_mut().enumerate() {
                let n = 17 + (seed * 23 + sm * 7) % 180;
                p.push(Instr::Load {
                    tensor: a,
                    global: View {
                        base: sm * 256 + seed,
                        rows: 1,
                        cols: n,
                        row_stride: 0,
                        col_stride: if seed % 3 == 0 { 64 } else { 1 },
                    },
                    local: View::contiguous(0, 1, n),
                });
                p.push(Instr::Vector {
                    kind: Op::Square,
                    dst: 0,
                    len: n,
                    a: Arg::reg(0),
                    b: Arg::Imm { value: 0. },
                });
                p.push(Instr::Store {
                    tensor: b,
                    global: View::contiguous(sm * 4096, 1, n),
                    local: View::contiguous(0, 1, n),
                });
            }
            m.wave(w).unwrap();
            reports.push(m.report());
        }
        assert_eq!(
            reports[0].stats.cycles, reports[1].stats.cycles,
            "seed {seed}"
        );
        assert_eq!(reports[0].stats.hbm_reads, reports[1].stats.hbm_reads);
        assert_eq!(
            reports[0].stats.hbm_queue_full_sm_cycles,
            reports[1].stats.hbm_queue_full_sm_cycles
        );
        assert!((reports[0].short_power_w - reports[1].short_power_w).abs() < 1e-8);
        assert!((reports[0].long_power_w - reports[1].long_power_w).abs() < 1e-8);
    }
}

#[test]
fn masked_store_preserves_other_words_and_pays_full_line() {
    let mut m = machine();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        load(a, 0, 16),
        Instr::Fill {
            dst: 0,
            len: 2,
            value: 9.,
        },
        Instr::Store {
            tensor: a,
            global: View::contiguous(3, 1, 2),
            local: View::contiguous(0, 1, 2),
        },
    ];
    m.wave(w).unwrap();
    assert_eq!(m.timing.stats.hbm_reads, 1);
    assert_eq!(m.timing.stats.hbm_writes, 1);
    assert_eq!(m.timing.stats.direction_switches, 1);
    assert_eq!(m.timing.stats.noc_bytes, 152);
    for (i, &v) in m.tensors[a].data.iter().enumerate() {
        assert_eq!(v, if i == 3 || i == 4 { 9. } else { 1. });
    }
}
#[test]
fn explicit_reduction_scratch_and_partial_alias_validation() {
    let mut m = machine();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 16,
            value: 1.,
        },
        Instr::Reduce {
            dst: 32,
            src: 0,
            len: 16,
            max: false,
            scratch: 8,
        },
    ];
    assert!(m.wave(w).unwrap_err().contains("overlap"));
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 17,
            value: 1.,
        },
        Instr::Vector {
            kind: Op::Square,
            dst: 1,
            len: 16,
            a: Arg::reg(0),
            b: Arg::Imm { value: 0. },
        },
    ];
    assert!(m.wave(w).unwrap_err().contains("alias"));
}
#[test]
fn requesting_report_does_not_advance_machine_into_idle_tail() {
    let mut m = machine();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0].push(load(a, 0, 16));
    m.wave(w.clone()).unwrap();
    let before = m.report().stats.cycles;
    m.wave(w).unwrap();
    assert!(m.report().stats.cycles > before);
}

#[test]
fn minimum_rf_and_large_ffn_fit_explicit_working_sets() {
    let h = Hardware {
        sms: 4,
        rf_kib: 32,
        rf_tier: 1,
        sh_kib: 0,
        sh_banks: 0,
        sh_tc_bw: 0,
        vector: 16,
        sfu: 2,
        p: 16,
        q: 16,
        hbm_channels: 8,
        hbm_queue: 64,
        sm_noc: 32,
        global_noc: 128,
    };
    let f = Fixture::new(
        Shape {
            layers: 1,
            d: 8,
            heads: 1,
            f: 4096,
            prefill: 2,
            history: 3,
            steps: 1,
        },
        11,
    )
    .unwrap();
    for decode in [false, true] {
        let mut m = Machine::new(h.clone(), true).unwrap();
        let actual = baseline::run(&mut m, &f, decode, 32).unwrap();
        reference::compare(&actual, &reference::run(&f, decode)).unwrap();
        assert!(m.timing.stats.max_write_staged <= 2);
    }
}

#[cfg(feature = "compact")]
#[test]
fn repeated_group_writes_still_conflict_with_another_groups_read() {
    let mut m = machine();
    let tensor = m.alloc("scratch", vec![1.0; 4]).unwrap();
    let mut wave = vec![vec![]; 8];
    wave[0] = vec![
        Instr::Store {
            tensor,
            global: View::contiguous(0, 1, 1),
            local: View::contiguous(0, 1, 1),
        };
        262_145
    ];
    wave[1].push(load(tensor, 0, 1));
    assert!(m.wave(wave).unwrap_err().contains("race"));
}
