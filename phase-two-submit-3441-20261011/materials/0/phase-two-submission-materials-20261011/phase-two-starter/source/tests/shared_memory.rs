use vnext_sim::{
    baseline::{self, Builder, Fixture, FixtureProfile, Matrix, Policy, Shape},
    hardware::Hardware,
    isa::{Instr, View},
    machine::Machine,
    reference,
};
fn hardware(banks: usize) -> Hardware {
    Hardware {
        sh_kib: 64,
        sh_banks: banks,
        ..Hardware::default()
    }
}
fn run_transfer(stride: usize, banks: usize, cycle: bool) -> Machine {
    let mut m = Machine::new(hardware(banks), true).unwrap();
    m.timing.cycle_reference = cycle;
    let a = m.alloc("a", (0..64).map(|x| x as f32).collect()).unwrap();
    let b = m.empty("b", 64).unwrap();
    let mut wave = vec![vec![]; 8];
    for (sm, program) in wave.iter_mut().enumerate().take(4) {
        let shared = View {
            base: 0,
            rows: 16,
            cols: 1,
            row_stride: stride,
            col_stride: 1,
        };
        let rf = View::contiguous(0, 16, 1);
        *program = vec![
            Instr::LoadShared {
                tensor: a,
                global: View::contiguous(sm * 16, 16, 1),
                local: shared,
            },
            Instr::SharedRead { shared, local: rf },
            Instr::SharedWrite { shared, local: rf },
            Instr::StoreShared {
                tensor: b,
                global: View::contiguous(sm * 16, 16, 1),
                local: shared,
            },
        ];
    }
    m.wave(wave).unwrap();
    assert_eq!(m.tensors[a].data, m.tensors[b].data);
    m
}
#[test]
fn shared_line_hand_cost_and_area() {
    let h = hardware(16);
    assert!(
        (h.area() - Hardware::default().area() - 8. * (0.004 * 64. + 0.015 * 16.)).abs() < 1e-10
    );
    let mut m = Machine::new(h, true).unwrap();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0].push(Instr::LoadShared {
        tensor: a,
        global: View::contiguous(0, 1, 16),
        local: View::contiguous(0, 1, 16),
    });
    m.wave(w).unwrap();
    // RF's 1+2 cycle reception is replaced with one SH bank cycle +6 latency.
    assert_eq!(m.timing.stats.cycles, 281);
    assert_eq!(m.timing.stats.rf_bytes, 0);
    assert_eq!(m.timing.stats.sh_bytes, 64);
    let expected = 8. + 20. + 8. + 2. + 8. + 64. * 150. + 72. * 2. + 64. * 0.8 + 8. * 8.;
    assert!((m.timing.power.dynamic_pj - expected).abs() < 1e-7);
}
#[test]
fn conflicts_and_return_backpressure_match_cycle_scheduler() {
    for banks in [16, 32, 64] {
        for stride in [1, 16, 17, 32, 64] {
            let mut a = run_transfer(stride, banks, false);
            let mut b = run_transfer(stride, banks, true);
            assert_eq!(
                serde_json::to_value(a.report()).unwrap(),
                serde_json::to_value(b.report()).unwrap()
            );
            let cycles = 16usize.div_ceil(banks / gcd(banks, stride));
            assert_eq!(
                a.timing.stats.sh_bank_stall_cycles,
                (4 * 4 * (cycles - 1)) as u64
            );
        }
    }
}
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
#[test]
fn same_word_read_broadcasts_and_rf_receives_every_lane() {
    let mut m = Machine::new(hardware(16), true).unwrap();
    let a = m.alloc("a", vec![7.]).unwrap();
    let b = m.empty("b", 16).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::LoadShared {
            tensor: a,
            global: View::contiguous(0, 1, 1),
            local: View::contiguous(0, 1, 1),
        },
        Instr::SharedRead {
            shared: View {
                base: 0,
                rows: 1,
                cols: 16,
                row_stride: 0,
                col_stride: 0,
            },
            local: View::contiguous(0, 1, 16),
        },
        Instr::Store {
            tensor: b,
            global: View::contiguous(0, 1, 16),
            local: View::contiguous(0, 1, 16),
        },
    ];
    m.wave(w).unwrap();
    assert_eq!(m.tensors[b].data, vec![7.; 16]);
    assert_eq!(m.timing.stats.sh_bytes, 8); // initial write and one broadcast read
    assert_eq!(m.timing.stats.rf_bytes, 128); // 16 RF writes and 16 DMA-source reads
}
#[test]
fn shared_lifetime_and_capacity_are_checked() {
    let mut m = Machine::new(hardware(16), true).unwrap();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::LoadShared {
        tensor: a,
        global: View::contiguous(0, 1, 16),
        local: View::contiguous(0, 1, 16),
    }];
    m.wave(w).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::SharedRead {
        shared: View::contiguous(0, 1, 16),
        local: View::contiguous(0, 1, 16),
    }];
    assert!(m.wave(w.clone()).is_err());
    assert!(!m.report().execution_valid);
    let mut no_sh = Machine::new(Hardware::default(), true).unwrap();
    assert!(no_sh.wave(w).is_err());
    assert!(
        Hardware {
            sh_kib: 0,
            sh_banks: 16,
            ..Hardware::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        Hardware {
            sh_kib: 64,
            sh_banks: 0,
            ..Hardware::default()
        }
        .validate()
        .is_err()
    );
    let mut m = Machine::new(hardware(16), true).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::SharedWrite {
        shared: View::contiguous(64 * 256, 1, 1),
        local: View::contiguous(0, 1, 1),
    }];
    assert!(m.wave(w).is_err());
}
#[test]
fn shared_row_reuse_reduces_hbm_bytes_by_the_expected_amount() {
    let mut bytes = vec![];
    for reuse in [0, 4] {
        let mut m = Machine::new(hardware(16), true).unwrap();
        let a = m.alloc("a", vec![1.; 128 * 64]).unwrap();
        let b = m.alloc("b", vec![1.; 64 * 64]).unwrap();
        let c = m.empty("c", 128 * 64).unwrap();
        let p = Policy {
            m: 16,
            n: 16,
            k: 16,
            sh_rows: reuse,
            ..Policy::square(16)
        };
        Builder {
            machine: &mut m,
            policy: p,
        }
        .gemm(
            Matrix::plain(a, 128, 64),
            Matrix::plain(b, 64, 64),
            Matrix::plain(c, 128, 64),
        )
        .unwrap();
        assert!(m.tensors[c].data.iter().all(|x| *x == 64.));
        bytes.push(m.timing.stats.valid_dma_bytes);
    }
    assert_eq!(bytes, vec![294912, 196608]);
}
#[test]
fn shared_complete_transformer_with_tails_matches_reference() {
    let f = Fixture::new_profile(
        Shape {
            layers: 2,
            d: 48,
            heads: 3,
            f: 97,
            prefill: 11,
            history: 13,
            steps: 2,
        },
        37,
        FixtureProfile::AttentionStress,
    )
    .unwrap();
    for decode in [false, true] {
        let expected = reference::run(&f, decode);
        let p = Policy {
            m: 3,
            n: 13,
            k: 17,
            sh_rows: 4,
            head_group: 2,
            persistent_keys: true,
            pack_transpose: true,
            ..Policy::square(3)
        };
        let mut reports = vec![];
        for cycle in [false, true] {
            let mut m = Machine::new(hardware(16), true).unwrap();
            m.timing.cycle_reference = cycle;
            let actual = baseline::run_policy(&mut m, &f, decode, p).unwrap();
            reference::compare(&actual, &expected).unwrap();
            reports.push(serde_json::to_value(m.report()).unwrap());
        }
        assert_eq!(reports[0], reports[1]);
    }
}

// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn local_pipeline_has_bounded_slots_and_pays_full_energy() {
    for chunks in [1usize, 8, 9, 32, 256] {
        let mut reports = vec![];
        for copy in [false, true] {
            let mut m = Machine::new(hardware(16), true).unwrap();
            let a = m.alloc("a", vec![1.; 4096]).unwrap();
            let mut w = vec![vec![]; 8];
            w[0].push(Instr::LoadShared {
                tensor: a,
                global: View::contiguous(0, 1, 4096),
                local: View::contiguous(0, 1, 4096),
            });
            if copy {
                w[0].push(Instr::SharedRead {
                    shared: View::contiguous(0, 1, chunks * 16),
                    local: View::contiguous(0, 1, chunks * 16),
                });
            }
            m.wave(w).unwrap();
            reports.push((
                m.timing.stats.cycles,
                m.timing.power.dynamic_pj,
                m.timing.stats.max_sh_copy_inflight,
            ));
        }
        // Every batch has 1+6 source and 1+2 destination cycles. Eight
        // completion-reserved slots yield eight issues every ten cycles.
        let duration = 7 + ((chunks - 1) / 8) * 10 + (chunks - 1) % 8 + 10;
        assert_eq!(reports[1].0 - reports[0].0, duration as u64);
        let energy = 8. + 20. + (chunks * 16) as f64 * 0.5 + (chunks * 64) as f64 * (0.8 + 0.462);
        assert!((reports[1].1 - reports[0].1 - energy).abs() < 1e-6);
        assert_eq!(reports[1].2, chunks.min(8));
    }
}

#[test]
fn shared_is_sm_private_and_static_rejection_does_not_poison() {
    let mut m = Machine::new(hardware(16), true).unwrap();
    let a = m.alloc("a", vec![1.; 16]).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::LoadShared {
        tensor: a,
        global: View::contiguous(0, 1, 16),
        local: View::contiguous(64 * 256, 1, 16),
    }];
    assert!(m.wave(w).is_err());
    assert!(m.report().execution_valid);
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::LoadShared {
        tensor: a,
        global: View::contiguous(0, 1, 16),
        local: View::contiguous(0, 1, 16),
    }];
    w[1] = vec![Instr::SharedRead {
        shared: View::contiguous(0, 1, 16),
        local: View::contiguous(0, 1, 16),
    }];
    assert!(m.wave(w).is_err());
    assert!(!m.report().execution_valid);
}

// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn direct_shared_mma_pays_interface_bandwidth_and_matches_rf_result() {
    let mut cycles = vec![];
    for bw in [32, 64, 128] {
        let h = Hardware {
            p: 8,
            q: 16,
            sh_tc_bw: bw,
            ..hardware(16)
        };
        let mut m = Machine::new(h.clone(), true).unwrap();
        let a = m.alloc("a", vec![1.; 8 * 32]).unwrap();
        let b = m.alloc("b", vec![2.; 32 * 16]).unwrap();
        let c = m.empty("c", 8 * 16).unwrap();
        let mut w = vec![vec![]; 8];
        w[0] = vec![
            Instr::Load {
                tensor: a,
                global: View::contiguous(0, 8, 32),
                local: View::contiguous(128, 8, 32),
            },
            Instr::LoadShared {
                tensor: b,
                global: View::contiguous(0, 32, 16),
                local: View::contiguous(0, 32, 16),
            },
            Instr::Fill {
                dst: 0,
                len: 128,
                value: 0.,
            },
            Instr::MmaShared {
                a: 128,
                b: View::contiguous(0, 32, 16),
                c: 0,
                m: 8,
                n: 16,
                k: 32,
            },
            Instr::Store {
                tensor: c,
                global: View::contiguous(0, 8, 16),
                local: View::contiguous(0, 8, 16),
            },
        ];
        let mut control = Machine::new(h.clone(), true).unwrap();
        control.alloc("a", vec![1.; 8 * 32]).unwrap();
        control.alloc("b", vec![2.; 32 * 16]).unwrap();
        control.empty("c", 8 * 16).unwrap();
        let mut without_mma = w.clone();
        without_mma[0].remove(3);
        control.wave(without_mma).unwrap();
        m.wave(w).unwrap();
        let expected_mma = 8. + 2048. * 0.462 + 2048. * 0.8 + 2048. * 0.2 + 4096. * 3.;
        assert!(
            (m.timing.power.dynamic_pj - control.timing.power.dynamic_pj - expected_mma).abs()
                < 1e-6
        );
        // Issue + accumulator read + input pipeline + K service + array drain + write.
        assert_eq!(
            m.timing.stats.cycles - control.timing.stats.cycles,
            1 + 4 + 10 + if bw == 32 { 64 } else { 32 } + 22 + 6
        );
        assert!(m.tensors[c].data.iter().all(|x| *x == 64.));
        assert_eq!(m.timing.stats.sh_tc_bytes, 2048);
        assert_eq!(
            m.timing.stats.sh_tc_stall_cycles,
            if bw == 32 { 32 } else { 0 }
        );
        // Load A + fill C + read/write C for MMA + A inputs + store C.
        assert_eq!(m.timing.stats.rf_bytes, 4096);
        cycles.push(m.timing.stats.cycles);
        assert!(
            (h.area()
                - Hardware {
                    sh_tc_bw: 0,
                    ..h.clone()
                }
                .area()
                - 8. * (0.28 + 0.004 * bw as f64))
                .abs()
                < 1e-10
        );
    }
    assert_eq!(cycles[0], cycles[1] + 32);
    assert_eq!(cycles[1], cycles[2]);
}

#[test]
fn direct_shared_complete_transformer_and_estimate_agree() {
    let f = Fixture::new_profile(
        Shape {
            layers: 2,
            d: 48,
            heads: 3,
            f: 97,
            prefill: 11,
            history: 13,
            steps: 2,
        },
        19,
        FixtureProfile::AttentionStress,
    )
    .unwrap();
    let p = Policy {
        m: 3,
        n: 13,
        k: 17,
        sh_rows: 4,
        sh_direct: true,
        head_group: 2,
        persistent_keys: true,
        pack_transpose: true,
        ..Policy::square(3)
    };
    let expected = reference::run(&f, false);
    let mut reports = vec![];
    for (functional, cycle) in [(true, false), (true, true), (false, false)] {
        let mut m = Machine::new(
            Hardware {
                sh_tc_bw: 32,
                ..hardware(16)
            },
            functional,
        )
        .unwrap();
        m.timing.cycle_reference = cycle;
        let actual = baseline::run_policy(&mut m, &f, false, p).unwrap();
        if functional {
            reference::compare(&actual, &expected).unwrap();
        }
        let mut r = serde_json::to_value(m.report()).unwrap();
        r.as_object_mut().unwrap().remove("functional");
        reports.push(r);
    }
    assert_eq!(reports[0], reports[1]);
    assert_eq!(reports[0], reports[2]);
}
